import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from models.layers import DecoderLayer_G, DecoderLayer_D
from models.sublayers import LinearWithChannel_2, PositionalEncoder


class Decoder_G(nn.Module):
    def __init__(self, n_layers=4, d_model=128, d_inner_scale=4, n_head=4, 
                d_k=32, d_v=32, dropout=0.1, scale_emb=False):
        super().__init__()
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.decoder_stack_1 = nn.ModuleList([DecoderLayer_G(d_model=d_model, 
                                                        d_inner=d_inner_scale * d_model,
                                                        n_head=n_head, 
                                                        d_k=d_k, 
                                                        d_v=d_v, 
                                                        dropout=dropout,
                                                        i=i)
                                                        for i in range(4)])
        
        self.fc = nn.Linear(d_model,d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pe = PositionalEncoder(d_model)
        self.decoder_stack_2 = nn.ModuleList([DecoderLayer_G(d_model=d_model, 
                                                        d_inner=d_inner_scale * d_model,
                                                        n_head=n_head, 
                                                        d_k=d_k, 
                                                        d_v=d_v, 
                                                        dropout=dropout,
                                                        i=i)
                                                        for i in range(4,8)])
        
    def forward(self, dec_inpput, enc_output, enc_mask):
        dec_output = dec_inpput
        for dec_layer in self.decoder_stack_1:
            dec_output, attn = dec_layer(dec_output, enc_output, enc_mask.unsqueeze(-2)) 
        #dec_output = self.fc(dec_output)
        dec_output = dec_output.repeat(1,24,1)
        dec_output = self.pe(dec_output)
        dec_output = self.fc(dec_output)
        for dec_layer in self.decoder_stack_2:
            dec_output, _ = dec_layer(dec_output, enc_output, enc_mask.unsqueeze(-2))    
        return dec_output, attn


class Decoder_D(nn.Module):
    def __init__(self, n_layers=4, d_model=128, d_inner_scale=4, n_head=4, 
                d_k=32, d_v=32, dropout=0.1, scale_emb=False, G=False):
        super().__init__()
        self.G = G
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.decoder_stack = nn.ModuleList([DecoderLayer_D(d_model=d_model, 
                                                        d_inner=d_inner_scale * d_model,
                                                        n_head=n_head, 
                                                        d_k=d_k, 
                                                        d_v=d_v, 
                                                        dropout=dropout,
                                                        i=i)
                                                        for i in range(4)])
        
    def forward(self, dec_inpput, enc_output, enc_mask):
        dec_output = dec_inpput
        for dec_layer in self.decoder_stack:
            dec_output, attn = dec_layer(dec_output, enc_output, enc_mask.unsqueeze(-2))     
        return dec_output, attn
