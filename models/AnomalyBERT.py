import torch
import torch.nn as nn
from layers.Embed import SinusoidalPositionalEncoding, AbsolutePositionEmbedding
from layers.AnomalyBERT_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import RelativeMultiHeadAttentionLayer, PositionWiseFeedForwardLayer


class Model(nn.Module):
    def __init__(self, args):
        """
        """
        super(Model, self).__init__()

        self.num_vars = args.n_vars
        self.d_model = args.d_model
        self.patch_size = args.patch_size
        self.input_encoder_len = args.input_encoder_len
        self.data_seq_len = self.patch_size * self.input_encoder_len

        hidden_dim = int(args.hidden_dim_rate * self.d_model)

        # Token embedding
        self.token_embedding = nn.Linear(self.num_vars * self.patch_size, self.d_model)
        nn.init.xavier_uniform_(self.token_embedding.weight)

        # Position embedding

        if args.positional_encoding.lower() == 'sinusoidal' or args.positional_encoding.lower() == 'sin':
            self.position_embedding = SinusoidalPositionalEncoding(self.d_model, self.input_encoder_len,
                                                                   args.dropout)
        elif args.positional_encoding.lower() == 'absolute' or args.positional_encoding.lower() == 'abs':
            self.position_embedding = AbsolutePositionEmbedding(self.d_model, self.input_encoder_len,
                                                                args.dropout)
        else:
            self.position_embedding = None

        # Transformer encoder
        encoder_layer = EncoderLayer(
            attention_layer=RelativeMultiHeadAttentionLayer(self.d_model, args.n_heads, self.input_encoder_len,
                                                            args.relative_position_embedding),
            feed_forward_layer=PositionWiseFeedForwardLayer(self.d_model, hidden_dim, args.dropout),
            norm_layer=nn.LayerNorm(self.d_model, eps=1e-6),
            dropout=args.dropout)
        self.encoder = Encoder(
            encoder_layer=encoder_layer,
            n_layer=args.e_layers
        )

        # MLP layers for output prediction
        self.mlp_layers = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, args.dim_output * self.patch_size)
        )
        nn.init.xavier_uniform_(self.mlp_layers[0].weight)
        nn.init.zeros_(self.mlp_layers[0].bias)
        nn.init.xavier_uniform_(self.mlp_layers[2].weight)
        nn.init.zeros_(self.mlp_layers[2].bias)

        # BERT-specific: masked token prediction head
        self.masked_token_pred = nn.Linear(self.d_model, self.d_model)
        nn.init.xavier_uniform_(self.masked_token_pred.weight)
        nn.init.zeros_(self.masked_token_pred.bias)

    def forward(self, x, mask=None):
        """
        <input info>
        x : (n_batch, max_seq_len, num_vars)
        mask : (n_batch, max_seq_len) optional mask for masked language modeling
        """
        n_batch = x.shape[0]

        # Apply token embedding
        embedded_out = self.token_embedding(
            x.view(n_batch, self.input_encoder_len, self.patch_size, -1).view(n_batch, self.input_encoder_len,
                                                                              -1))

        # Add positional embedding
        if self.position_embedding is not None:
            position_ids = torch.arange(self.input_encoder_len, device=x.device).unsqueeze(0).expand(n_batch, -1)
            embedded_out = self.position_embedding(position_ids)

        # Pass through transformer encoder
        encoder_out = self.encoder(embedded_out)

        # MLP layers for general feature extraction
        output = self.mlp_layers(encoder_out)

        # Masked token prediction (if mask is provided)
        if mask is not None:
            masked_token_output = self.masked_token_pred(encoder_out)
            return output, masked_token_output

        return output.view(n_batch, self.input_encoder_len, self.patch_size, -1).view(n_batch, self.data_seq_len, -1)
