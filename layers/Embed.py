import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np
import einops


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_seq_len=512, dropout=0.1):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout)

        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        denominators = torch.exp(torch.arange(0, d_embed, 2) * (np.log(0.0001) / d_embed)).unsqueeze(0)
        encoding_matrix = torch.matmul(positions, denominators)

        encoding = torch.empty(1, max_seq_len, d_embed)
        encoding[0, :, 0::2] = torch.sin(encoding_matrix)
        encoding[0, :, 1::2] = torch.cos(encoding_matrix[:, :(d_embed // 2)])

        self.register_buffer('encoding', encoding)

    def forward(self, x):
        """
        <input info>
        x : (n_batch, n_token, d_embed) == (*, max_seq_len, d_embed) (default)
        """
        return self.dropout_layer(x + self.encoding)


# Absolute position embedding
class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, d_embed, max_seq_len=512, dropout=0.1):
        super(AbsolutePositionEmbedding, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_embed))
        trunc_normal_(self.embedding, std=.02)

    def forward(self, x):
        """
        <input info>
        x : (n_batch, n_token, d_embed) == (*, max_seq_len, d_embed) (default)
        """
        return self.dropout_layer(x + self.embedding)


# Learnable positional encoding
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_seq_len=512, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.positional_encoding = nn.Embedding(max_seq_len, d_embed)

        def forward(self, x):
            """
            <input info>
            x : (n_batch, n_token, d_embed) == (*, max_seq_len, d_embed) (default)
            """
            positions = torch.arange(x.size(1), device=x.device)
            pos_embeddings = self.position_embeddings(positions)
            return self.dropout_layer(x + pos_embeddings)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in,
                                   out_channels=d_model,
                                   kernel_size=3,
                                   padding=1,
                                   padding_mode='circular',
                                   bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, c_in, d_embed, patch_size, embed_type='fixed', dropout=0.1):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        # Position Embedding: sin/cos or learnable
        if embed_type.lower() == 'fixed':
            self.position_embedding = SinusoidalPositionalEncoding(d_embed,
                                                                   c_in,
                                                                   dropout)
        elif embed_type.lower() in ['absolute', 'abs']:
            self.position_embedding = AbsolutePositionEmbedding(d_embed,
                                                                c_in,
                                                                dropout)
        elif embed_type.lower() in ['learnable', 'learn']:
            self.position_embedding = LearnablePositionalEncoding(d_embed,
                                                                  c_in,
                                                                  dropout)
        else:
            self.position_embedding = nn.Identity()

        self.value_embedding = TokenEmbedding(c_in, d_embed)

    def forward(self, x):
        try:
            patches_einops = einops.rearrange(x, 'b (n p) h -> b n p h', p=self.patch_size)
            x = self.value_embedding(patches_einops)
        except ImportError:
            print("\nEinops not available")

        return self.position_embedding(x)


class RandomMasking(nn.Module):
    def __init__(self, mask_ratio=0.15):
        super(RandomMasking, self).__init__()
        self.mask_ratio = mask_ratio

    def forward(self, xb):
        # xb = xb.permute(0, 2, 1, 3)  # [bs x num_patch x n_vars x patch_len]
        bs, L, nvars, D = xb.shape
        x = xb.clone()
        len_keep = int(L * (1 - self.mask_ratio))

        # Generate noise only for patches (not for each variable)
        noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L

        # Sort noise and get indices
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L]

        # Expand indices for all variables
        ids_shuffle = ids_shuffle.unsqueeze(-1).expand(-1, -1, nvars)  # [bs x L x nvars]
        ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, nvars)  # [bs x L x nvars]

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep, :]  # ids_keep: [bs x len_keep x nvars]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))

        # Create removed x
        x_removed = torch.zeros(bs, L - len_keep, nvars, D, device=xb.device)

        # Combine kept and removed parts
        x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len]

        # Restore the original order with masked values
        x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, D))

        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([bs, L, nvars], device=x.device)
        mask[:, :len_keep, :] = 0

        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch x nvars]
        return x_masked, mask


class PatchEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_embed, patch_size, dropout=0.1, mask=False, mask_ratio=0.15):
        super(PatchEmbedding_wo_pos, self).__init__()
        self.patch_size = patch_size
        self.stride = patch_size
        padding = 0
        self.mask = mask
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))  # padding to the end of this sequence
        if mask:
            self.random_masking = RandomMasking(mask_ratio)

        self.value_embedding = TokenEmbedding(c_in, d_embed)  # 1DConv
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [bs, input_len, n_vars]
        """
        mask_map = None
        x = self.padding_patch_layer(x)
        x = x.permute(0, 2, 1).unfold(dimension=-1, size=self.patch_size,
                                      step=self.stride)  # [Batch, n_vars, patch_num, patch_len]
        x = x.permute(0, 2, 1, 3)  # [Batch, patch_num, n_vars, patch_len]
        y = x.permute(0, 1, 3, 2)  # [Batch, patch_num, patch_len, n_vars]
        if self.mask:
            x_masked, mask_map = self.random_masking(x)  # [bs x num_patch x nvars x patch_len]
            x = x_masked
        x = einops.rearrange(x, 'b n v p -> b n (p v)', p=self.patch_size)  # [Batch, patch_num, patch_len * n_vars]
        x = self.value_embedding(x)  # [bs, num_patch, d_emb]
        return self.dropout(x), y, mask_map  # [bs, num_patch, n_vars, patch_len] and [bs, num_patch, n_vars]

# embedded_out = self.token_embedding(
#             x.view(n_batch, self.input_encoder_length, self.patch_size, -1).
#             view(n_batch, self.input_encoder_length,-1))
