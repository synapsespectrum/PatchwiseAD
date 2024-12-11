import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.functions import clone_layer


class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, feed_forward_layer, norm_layer, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention_layer = attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.norm_layers = clone_layer(norm_layer, 2)
        self.dropout_layer = nn.Dropout(p=dropout)

        for p in self.attention_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        for p in self.feed_forward_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, x):
        out1 = self.norm_layers[0](x)  # Layer norm first
        out1 = self.attention_layer(out1)
        out1 = self.dropout_layer(out1) + x

        out2 = self.norm_layers[1](out1)
        out2 = self.feed_forward_layer(out2)
        return self.dropout_layer(out2) + out1


class Encoder(nn.Module):
    def __init__(self, encoder_layer, n_layer):
        super(Encoder, self).__init__()
        self.encoder_layers = clone_layer(encoder_layer, n_layer)

    def forward(self, x):
        """
        <input>
        x : (n_batch, n_token, d_embed)
        """
        position_vector = None
        
        out = x

        for layer in self.encoder_layers:
            out = layer(out)

        return out
