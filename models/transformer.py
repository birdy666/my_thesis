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
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.decoder_stack = nn.ModuleList([DecoderLayer(d_model=d_model, 
                                                        d_inner=d_inner_scale * d_model,
                                                        n_head=n_head, 
                                                        d_k=d_k, 
                                                        d_v=d_v, 
                                                        dropout=dropout,
                                                        selfatt= not(G and i==0))
                                                        for i in range(n_layers)])
        
    def forward(self, dec_inpput, enc_output, enc_mask):
        dec_output = dec_inpput
        for dec_layer in self.decoder_stack:
            dec_output = dec_layer(dec_output, enc_output, enc_mask.unsqueeze(-2))         
        return dec_output
