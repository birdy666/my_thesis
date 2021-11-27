import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from models.layers import EncoderLayer, DecoderLayer, LinearDecoderLayer
from models.sublayers import PositionalEncoder, LinearWithChannel, MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, n_layers=4, d_model=144, d_inner_scale=4, n_head=8, 
                d_k=32, d_v=32, dropout=0.1, scale_emb=False):
        super().__init__()
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.encoder_stack = nn.ModuleList([
                                            EncoderLayer(d_model=d_model, 
                                                        d_inner=d_inner_scale * d_model,
                                                        n_head=n_head, 
                                                        d_k=d_k, 
                                                        d_v=d_v, 
                                                        dropout=dropout)
                                                        for _ in range(n_layers)])

    def forward(self, enc_input, enc_mask=None):
        if enc_mask is not None:
            enc_mask= enc_mask.unsqueeze(-2)
        enc_output = self.dropout(self.pe(enc_input))
        enc_output = self.layer_norm(enc_output)
         
        for enc_layer in self.encoder_stack:         
            enc_output = enc_layer(enc_output , slf_attn_mask=enc_mask)  
        return enc_output

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(self, n_layers=4, d_model=144, d_inner_scale=2, n_head=8, 
                d_k=32, d_v=32, dropout=0.1, scale_emb=False):
        super().__init__()
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.decoder_stack = nn.ModuleList([DecoderLayer(d_model=d_model, 
                                                        d_inner=d_inner_scale * d_model,
                                                        n_head=n_head, 
                                                        d_k=d_k, 
                                                        d_v=d_v, 
                                                        dropout=dropout)
                                                        for _ in range(n_layers)])

    def forward(self, enc_output, enc_mask, dec_input, dec_mask=None):
        if dec_mask is not None:
            dec_mask= dec_mask.unsqueeze(-2)
        dec_output = self.dropout(self.pe(dec_input))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.decoder_stack:
            dec_output = dec_layer(dec_output, enc_output, slf_attn_mask=dec_mask, dec_enc_attn_mask=enc_mask.unsqueeze(-2)) 
        return dec_output

class LinearDecoder(nn.Module):
    def __init__(self, d_vec, d=False):
        super().__init__()
        self.pe = PositionalEncoder(d_vec, dropout=0.1)
        self.ln_dec_layer_1 =  LinearDecoderLayer(d_vec, d_vec//2, s=4) # 144 -> 72
        self.slf_attn_1 = MultiHeadAttention(n_head=4, d_model=d_vec//2, d_k=d_vec//2//4, d_v=d_vec//2//4, dropout=0.1)
        self.ln_dec_layer_2 =  LinearDecoderLayer(d_vec//2, d_vec//3, s=4) # 72 -> 48
        self.slf_attn_2 = MultiHeadAttention(n_head=4, d_model=d_vec//3, d_k=d_vec//3//4, d_v=d_vec//3//4, dropout=0.1)
        self.ln_dec_layer_3 =  LinearDecoderLayer(d_vec//3, d_vec//4, s=4) # 48 ->36
        self.slf_attn_3 = MultiHeadAttention(n_head=4, d_model=d_vec//4, d_k=d_vec//4//4, d_v=d_vec//4//4, dropout=0.1)
    
        self.ln_dec_layer_4 =  LinearDecoderLayer(d_vec//4, 3, s=4)


    def forward(self, x):
        x = self.pe(x)

        x = self.ln_dec_layer_1(x)
        x = self.slf_attn_1(x,x,x)

        x = self.ln_dec_layer_2(x)
        x = self.slf_attn_2(x,x,x)

        x = self.ln_dec_layer_3(x)
        x = self.slf_attn_3(x,x,x)

        x = self.ln_dec_layer_4(x)      
        return F.hardtanh(x, min_val=-math.pi+0.0000000001, max_val=math.pi-0.0000000001)
        
