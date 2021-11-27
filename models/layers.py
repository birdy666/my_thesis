
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

    def forward(self, enc_input, slf_attn_mask=None):
        x = enc_input + self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = x + self.pos_ffn(x)
        return enc_output

class DecoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_dec_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None): 
        x = dec_input + self.slf_attn(dec_input, dec_input, dec_input, slf_attn_mask)
        x = x + self.enc_dec_attn(x, enc_output, enc_output, mask=dec_enc_attn_mask)
        enc_dec_output = x + self.pos_ffn(x)
        return enc_dec_output


class LinearDecoderLayer(nn.Module):
    def __init__(self, d_in, d_out, s=4, dropout=0.1):
        super(LinearDecoderLayer, self).__init__()
        self.fcc_1 = LinearWithChannel(d_in, s*d_in, 24)
        self.fcc_2 = LinearWithChannel(s*d_in, d_in, 24)
        self.fcc_3 = LinearWithChannel(d_in, d_out, 24)
        #self.fc = nn.Linear(24*d_out, 24*d_out)
        self.dropout = nn.Dropout(dropout)        

    def forward(self, x):
        batch_size = x.size(0)
        x = self.fcc_1(x)
        x = self.fcc_2(x)
        x = self.fcc_3(x) #.view(batch_size,-1)
        #x = self.fc(x).view(batch_size,24,-1)

        return self.dropout(x)

