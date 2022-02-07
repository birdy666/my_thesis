import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from models.layers import EncoderLayer, DecoderLayer
from models.sublayers import PositionalEncoder

class Encoder(nn.Module):
    def __init__(self, n_layers=4, d_model=128, d_inner_scale=4, n_head=8, 
                d_k=32, d_v=32, dropout=0.1, scale_emb=False):
        super().__init__()
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.encoder_stack = nn.ModuleList([EncoderLayer(d_model=d_model, 
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

        for enc_layer in self.encoder_stack:         
            enc_output = enc_layer(enc_output , slf_attn_mask=enc_mask)  
        return enc_output

class Decoder(nn.Module):
    def __init__(self, n_layers=4, d_model=128, d_inner_scale=4, n_head=4, 
                d_k=32, d_v=32, dropout=0.1, scale_emb=False):
        super().__init__()
        self.decoder_stack = nn.ModuleList([DecoderLayer(d_model=d_model, 
                                                        d_inner=d_inner_scale * d_model,
                                                        n_head=n_head, 
                                                        d_k=d_k, 
                                                        d_v=d_v, 
                                                        dropout=dropout)
                                                         for _ in range(n_layers)])

    def forward(self, noise, enc_output, mask): 
        for dec_layer in self.decoder_stack:
            dec_output = dec_layer(noise, enc_output, mask)         
        return F.hardtanh(dec_output, min_val=-math.pi+0.0000000001, max_val=math.pi-0.0000000001)
