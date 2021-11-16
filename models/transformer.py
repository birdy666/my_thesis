import torch
import torch.nn as nn
import numpy as np
from models.layers import EncoderLayer, DecoderLayer


class Encoder(nn.Module):
    def __init__(self, n_layers=4, d_model=150, d_inner_scale=4, n_head=8, 
                d_k=32, d_v=32, dropout=0.1, scale_emb=False):
        super().__init__()
        self.scale_emb = scale_emb
        self.d_model = d_model
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

    def forward(self, enc_input, enc_mask):
        enc_output = enc_input        
        for enc_layer in self.encoder_stack:         
            enc_output = enc_layer(enc_output , slf_attn_mask=enc_mask.unsqueeze(-2))  
        return self.layer_norm(enc_output)

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(self, n_layers=4, d_model=150, d_inner_scale=2, n_head=8, 
                d_k=32, d_v=32, dropout=0.1, scale_emb=False):
        super().__init__()
        #self.position_enc = PositionalEncoding(cfg.D_WORD_VEC+cfg.NOISE_SIZE, n_position=cfg.N_POSITION_G)
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
        dec_output = dec_input
        for dec_layer in self.decoder_stack:
            dec_output = dec_layer(dec_output, enc_output, slf_attn_mask=dec_mask, dec_enc_attn_mask=enc_mask.unsqueeze(-2)) 
        return self.layer_norm(dec_output)

class Transformer(nn.Module):
    def __init__(self, enc_param, dec_param, fc_list):
        super().__init__()
        self.encoder = Encoder(n_layers=enc_param.n_layers, 
                                d_model=enc_param.d_model, 
                                d_inner_scale=enc_param.d_inner_scale, 
                                n_head=enc_param.n_head, 
                                d_k=enc_param.d_k, 
                                d_v=enc_param.d_v, 
                                dropout=enc_param.dropout, 
                                scale_emb=enc_param.scale_emb)

        self.decoder = Decoder(n_layers=dec_param.n_layers, 
                                d_model=dec_param.d_model, 
                                d_inner_scale=dec_param.d_inner_scale, 
                                n_head=dec_param.n_head, 
                                d_k=dec_param.d_k, 
                                d_v=dec_param.d_v, 
                                dropout=dec_param.dropout, 
                                scale_emb=dec_param.scale_emb)
        self.fcs = None
        if len(fc_list) > 0:
            layers = []
            for i in range(len(fc_list)-1):
                layers.append(nn.Linear(fc_list[i], fc_list[i+1]))
            self.fcs = nn.Sequential(*layers)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    def forward(self, enc_input, enc_mask, dec_input, dec_mask=None):
        # encoder output
        enc_output = self.encoder(enc_input, enc_mask)
        # decoder output
        dec_output = self.decoder(enc_output, enc_mask, dec_input, dec_mask=dec_mask)
        
        return dec_output