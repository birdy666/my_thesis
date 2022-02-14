import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from models.layers import DecoderLayer
from models.sublayers import PositionalEncoder

class Decoder(nn.Module):
    def __init__(self, n_layers=4, d_model=128, d_inner_scale=4, n_head=4, 
                d_k=32, d_v=32, dropout=0.1, scale_emb=False, G=False):
        super().__init__()
        self.pe_t = PositionalEncoder(d_model)        
        self.dropout_t = nn.Dropout(p=dropout)
        self.pe_r = PositionalEncoder(d_model)        
        self.dropout_r = nn.Dropout(p=dropout)
        self.decoder_stack = nn.ModuleList([DecoderLayer(d_model=d_model, 
                                                        d_inner=d_inner_scale * d_model,
                                                        n_head=n_head, 
                                                        d_k=d_k, 
                                                        d_v=d_v, 
                                                        dropout=dropout)
                                                        for _ in range(n_layers)])
        
    def forward(self, dec_inpput, enc_output, enc_mask):
        dec_output = self.dropout_r(self.pe_r(dec_inpput))
        enc_output = self.dropout_t(self.pe_t(enc_output))
        for dec_layer in self.decoder_stack:
            dec_output = dec_layer(dec_output, enc_output, enc_mask.unsqueeze(-2))         
        return dec_output
