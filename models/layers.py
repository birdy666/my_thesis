
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from models.sublayers import MultiHeadAttention, PositionwiseFeedForward, LinearWithChannel

"""class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, enc_input, enc_mask=None):
        x = self.slf_attn(enc_input, enc_input, enc_input, mask=enc_mask)
        x = self.layer_norm_1(x)
        x = self.pos_ffn(x)
        x = self.layer_norm_2(x)
        return x"""


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, selfatt=True):
        super(DecoderLayer, self).__init__()
        self.selfatt = selfatt
        if selfatt:
            self.slf_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout)
            self.layer_norm_1 = nn.LayerNorm((24, d_model), eps=1e-6)
            self.pos_ffn_1 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
            self.layer_norm_2 = nn.LayerNorm((24, d_model), eps=1e-6)

        self.enc_dec_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout)        
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)
        self.pos_ffn_2 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout) 
        self.layer_norm_4 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, dec_input, enc_output, enc_mask):
        x = dec_input
        if self.selfatt:
            x = self.slf_attn(x, enc_output, enc_output, mask=enc_mask)
            x = self.layer_norm_1(x)
            x = self.pos_ffn_1(x)
            x = self.layer_norm_2(x)

        x = self.enc_dec_attn(x, enc_output, enc_output, mask=enc_mask)
        x = self.layer_norm_3(x)
        x = self.pos_ffn_2(x)
        x = self.layer_norm_4(x)
        return x
"""

class DecoderLayerG(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, layer_index=0):
        super(DecoderLayerG, self).__init__()
        self.layer_index = layer_index
        if layer_index != 0:
            self.slf_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout)
            self.layer_norm_1 = nn.LayerNorm((24, d_model), eps=1e-6)
            self.pos_ffn_1 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
            self.layer_norm_2 = nn.LayerNorm((24, d_model), eps=1e-6)

        self.enc_dec_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)
        self.pos_ffn_2 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm_4 = nn.LayerNorm(d_model, eps=1e-6)
        

    def forward(self, dec_input, enc_output, enc_mask):
        x = dec_input

        if self.layer_index != 0:
            x = self.slf_attn(x, enc_output, enc_output, enc_mask)
            x = self.layer_norm_1(x)
            x = self.pos_ffn_1(x)  
            x = self.layer_norm_2(x)

        x = self.enc_dec_attn(x, enc_output, enc_output, enc_mask)
        x = self.layer_norm_3(x)
        x = self.pos_ffn_2(x)  
        x = self.layer_norm_4(x)
        return x"""

