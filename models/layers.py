
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from models.sublayers import LinearWithChannel, MultiHeadAttention, PositionwiseFeedForward, LinearWithChannel_2

class DecoderLayer_G(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, i=0):
        super(DecoderLayer_G, self).__init__()
        self.i = i
        if i > 3:
            self.slf_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout)
            self.pos_ffn_1 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
                     
            self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
            self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        else:
            self.enc_dec_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout)        
            self.pos_ffn_2 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        
            self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)
            self.layer_norm_4 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, dec_input, enc_output, enc_mask):
        x = dec_input
        if self.i > 3:
            x, attn = self.slf_attn(x, x, x)
            x = self.layer_norm_1(x)            
            x = self.pos_ffn_1(x)            
            x = self.layer_norm_2(x)
        else:
            x, attn = self.enc_dec_attn(x, enc_output, enc_output, mask=enc_mask)
            x = self.layer_norm_3(x)
            x = self.pos_ffn_2(x)        
            x = self.layer_norm_4(x)
        return x, attn

class DecoderLayer_D(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, i=0):
        super(DecoderLayer_D, self).__init__()
        self.enc_dec_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout)        
        self.pos_ffn_2 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_4 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, dec_input, enc_output, enc_mask):
        x = dec_input
        x, attn = self.enc_dec_attn(x, enc_output, enc_output, mask=enc_mask)
        x = self.layer_norm_3(x)
        x = self.pos_ffn_2(x)        
        x = self.layer_norm_4(x)
        return x, attn