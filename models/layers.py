
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        # square root of d_k 
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q , k.transpose(-2, -1)) / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        
        return output

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    
    """
    q = k = v = enc_input
    """
    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        score = self.attention(q, k, v, mask=mask)

        score_concat = score.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        output = self.fc(score_concat)
        return output

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ln_2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, enc_input, slf_attn_mask=None):
        x1 = self.ln_1(enc_input)
        x = enc_input + self.dropout_1(self.slf_attn(x1, x1, x1, mask=slf_attn_mask))
        x2 = self.ln_2(x)
        enc_output = x + self.dropout_2(self.pos_ffn(x2))
        return enc_output

class DecoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_dec_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ln_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ln_3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):   
        x1 = self.ln_1(dec_input)
        x = dec_input + self.dropout_1(self.slf_attn(x1, x1, x1, slf_attn_mask))
        x2 = self.ln_2(x)
        x = x + self.dropout_2(self.enc_dec_attn(x2, enc_output, enc_output, mask=dec_enc_attn_mask))
        x3 = self.ln_3(x)
        enc_dec_output = x + self.dropout_3(self.pos_ffn(x3))

        return enc_dec_output
