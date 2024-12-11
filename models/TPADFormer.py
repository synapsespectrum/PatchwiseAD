import torch
import torch.nn as nn
from layers.Embed import PatchEmbedding_wo_pos
from layers.AnomalyBERT_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import RelativeMultiHeadAttentionLayer, PositionWiseFeedForwardLayer
import einops


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, n_vars, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.n_vars = n_vars
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len * n_vars)

    def forward(self, x):
        x = self.linear(self.dropout(x))  # [bs, num_patch, nvars*patch_len]
        x = x.view(x.shape[0], x.shape[1], self.patch_len, self.n_vars)  # [Batch, patch_num, patch_len, n_vars]
        return x


class Model(nn.Module):
    def __init__(self, configs):
        """
        data_seq_len: the length of the input sequence
        input_encoder_length: the length of the input encoder
        patch_size: the size of the patch for patch embedding
        """
        super(Model, self).__init__()
        self.configs = configs
        self.n_vars = configs.n_vars  # number of variables
        self.d_model = configs.d_model
        self.patch_size = configs.patch_size
        self.input_encoder_len = configs.input_encoder_len
        self.stride = self.patch_size - configs.overlap
        self.data_seq_len = (self.input_encoder_len -1) * self.stride + self.patch_size
        hidden_dim = int(configs.hidden_dim_rate * self.d_model)

        # self.revin_layer = RevIN(configs.enc_in)  # enc_in is the number of input variables

        # patching and embedding
        # Position Embedding: sin/cos or none
        # Embedding: [::, n_vars * patch_size] -> [::, d_model]

        # Transformer Encoder
        self.encoder = Encoder(
            encoder_layer=EncoderLayer(
                attention_layer=RelativeMultiHeadAttentionLayer(self.d_model, configs.n_heads,
                                                                self.input_encoder_len,
                                                                configs.relative_position_embedding),
                feed_forward_layer=PositionWiseFeedForwardLayer(self.d_model,
                                                                hidden_dim,
                                                                configs.dropout),
                norm_layer=nn.LayerNorm(self.d_model, eps=1e-6),
                dropout=configs.dropout),
            n_layer=configs.e_layers)

        # Pretrain model or Downstream model
        if configs.is_pretrain:  # reconstruction the MASKED patches
            self.patch_embedding = PatchEmbedding_wo_pos(self.n_vars * self.patch_size,
                                                         self.d_model,
                                                         self.patch_size,
                                                         dropout=configs.dropout,
                                                         mask=True,
                                                         mask_ratio=configs.mask_ratio)
            self.head = PretrainHead(self.d_model, self.patch_size, self.n_vars, configs.dropout)
        else:  # fine-tuning
            self.patch_embedding = PatchEmbedding_wo_pos(self.n_vars * self.patch_size,
                                                         self.d_model,
                                                         self.patch_size,
                                                         configs.dropout)
            if configs.dataset == 'MSL': # avoid NAN happening in the output
                self.head = nn.Sequential(
                    nn.Linear(self.d_model, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, configs.dim_output * self.patch_size),
                    nn.Tanh()
                )
            else:
                self.head = nn.Sequential(
                    nn.Linear(self.d_model, hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.data_seq_len, configs.dim_output * self.patch_size)
                )
            nn.init.xavier_uniform_(self.head[0].weight)
            nn.init.zeros_(self.head[0].bias)
            nn.init.xavier_uniform_(self.head[2].weight)
            nn.init.zeros_(self.head[2].bias)
            if configs.pretrain_model_path is not None:
                self.transfer_weights()

    def forward(self, x):
        """
        x: tensor [bs x seq_len x nvars]
        """
        # x = self.revin_layer(x)
        x, y, mask_map = self.patch_embedding(x)  # [bs, num_patch, patch_len ,d_model]
        x = self.encoder(x)  # [bs, num_patch, d_model]
        x = self.head(x)
        if not self.configs.is_pretrain:
            x = einops.rearrange(x, 'b n p -> b (n p) 1')
            return x

        return x, y, mask_map

    def transfer_weights(self, exclude_head=True):
        print(f"Transfer weights from {self.configs.pretrain_model_path}...")
        pretrain_state = torch.load(self.configs.pretrain_model_path)
        # load embedding weights
        embedding_weight = pretrain_state['patch_embedding.value_embedding.tokenConv.weight']
        self.patch_embedding.value_embedding.tokenConv.weight.data.copy_(embedding_weight)
        print("Embedding weights successfully transferred!")
        # load encoder weight
        encoder_state = {k.replace('encoder.', ''): v
                         for k, v in pretrain_state.items()
                         if k.startswith('encoder.')}
        self.encoder.load_state_dict(encoder_state)
        print("Encoder weights successfully transferred!")
        print(f"weights from {self.configs.pretrain_model_path} successfully transferred!\n")
