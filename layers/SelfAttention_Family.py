import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from utils.functions import clone_layer


class RelativeMultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_embed, n_head, max_seq_len=512, relative_position_embedding=True):
        super(RelativeMultiHeadAttentionLayer, self).__init__()
        assert d_embed % n_head == 0  # Ckeck if d_model is divisible by n_head.

        self.d_embed = d_embed
        self.n_head = n_head
        self.d_k = d_embed // n_head
        self.scale = 1 / np.sqrt(self.d_k)

        self.word_fc_layers = clone_layer(nn.Linear(d_embed, d_embed), 3)
        self.output_fc_layer = nn.Linear(d_embed, d_embed)

        self.max_seq_len = max_seq_len
        self.relative_position_embedding = relative_position_embedding
        if relative_position_embedding:
            # Table of 1D relative position embedding
            self.relative_position_embedding_table = nn.Parameter(torch.zeros(2 * max_seq_len - 1, n_head))
            trunc_normal_(self.relative_position_embedding_table, std=.02)

            # Set 1D relative position embedding index.
            coords_h = np.arange(max_seq_len)
            coords_w = np.arange(max_seq_len - 1, -1, -1)
            coords = coords_h[:, None] + coords_w[None, :]
            self.relative_position_index = coords.flatten()

    def forward(self, x):
        """
        <input>
        x : (n_batch, n_token, d_embed)
        """
        n_batch = x.shape[0]
        device = x.device

        # Apply linear layers.
        query = self.word_fc_layers[0](x)
        key = self.word_fc_layers[1](x)
        value = self.word_fc_layers[2](x)

        # Split heads.
        query_out = query.view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)
        key_out = key.view(n_batch, -1, self.n_head, self.d_k).contiguous().permute(0, 2, 3, 1)
        value_out = value.view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)

        # Compute attention and concatenate matrices.
        scores = torch.matmul(query_out * self.scale, key_out)

        # Add relative position embedding
        if self.relative_position_embedding:
            position_embedding = self.relative_position_embedding_table[self.relative_position_index].view(
                self.max_seq_len, self.max_seq_len, -1)
            position_embedding = position_embedding.permute(2, 0, 1).contiguous().unsqueeze(0)
            scores = scores + position_embedding

        #         if masking_matrix != None:
        #             scores = scores + masking_matrix * (-1e9) # Add very small negative number to padding columns.
        probs = F.softmax(scores, dim=-1)
        attention_out = torch.matmul(probs, value_out)

        # Convert 4d tensor to proper 3d output tensor.
        attention_out = attention_out.transpose(1, 2).contiguous().view(n_batch, -1, self.d_embed)

        return self.output_fc_layer(attention_out)


class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, d_embed, d_ff, dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.first_fc_layer = nn.Linear(d_embed, d_ff)
        self.second_fc_layer = nn.Linear(d_ff, d_embed)
        self.activation_layer = nn.GELU()
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.first_fc_layer(x)
        out = self.dropout_layer(self.activation_layer(out))
        return self.second_fc_layer(out)
