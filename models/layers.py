
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from models.sublayers import MultiHeadAttention, PositionwiseFeedForward, LinearWithChannel

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        # we don't do layer norm in sublayer cause G_decoder don't need it
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, enc_input, slf_attn_mask=None):
        x = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        x = self.layer_norm_1(x)
        enc_output = self.pos_ffn(x)
        x = self.layer_norm_2(x)
        return enc_output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.enc_dec_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, enc_output, mask):
        x = self.enc_dec_attn(x, enc_output, enc_output, mask)
        x = self.layer_norm_1(x)
        x = self.pos_ffn(x)
        x = self.layer_norm_2(x)
        return x

