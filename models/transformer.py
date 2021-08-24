import torch
import torch.nn as nn
import numpy as np
from models.layers import PositionalEncoding, EncoderLayer, DecoderLayer


class Encoder(nn.Module):
    def __init__(self, n_layers=4, d_model=150, d_inner_scale=4, n_head=8, 
                d_k=32, d_v=32, dropout=0.1, scale_emb=False):
        super().__init__()
        self.scale_emb = scale_emb
        self.d_model = d_model
        """TODO 不了解"""
        self.position_enc = PositionalEncoding(d_model)
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
        # -- Forward
        if self.scale_emb:
            enc_input *= self.d_model ** 0.5
        """self.position_enc
        self.dropout
        self.layer_norm"""
        enc_output = enc_input        
        for enc_layer in self.encoder_stack:  
            """mask: (batch, text_len) ==> unsqueeze(-2) ==> (batch, 1, text_len)"""          
            enc_output = enc_layer(enc_output , slf_attn_mask=enc_mask.unsqueeze(-2))  
        return enc_output

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(self, n_layers=4, d_model=150, d_inner_scale=4, n_head=8, 
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

    def forward(self, enc_output, enc_mask, dec_input, dec_mask):
        # -- Forward
        #if self.cfg.SCALE_EMB_G:
        #    input_vec *= self.cfg.D_MODEL_LIST_G[0] ** 0.5
        #input_vec = self.dropout(self.position_enc(input_vec))
        """dec_input  = self.layer_norm(self.dropout(dec_input))"""
        dec_output = dec_input
        for dec_layer in self.decoder_stack:
            dec_output = dec_layer(dec_output, enc_output, slf_attn_mask=dec_mask, dec_enc_attn_mask=enc_mask.unsqueeze(-2)) 
        return dec_output


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, encoder, decoder, fc_list):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        layers = []
        for i in range(len(fc_list)-1):
            layers.append(nn.Linear(fc_list[i], fc_list[i+1]))
        self.fcs = nn.Sequential(*layers)

        """"for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) """

    def forward(self, enc_input, enc_mask, dec_input, dec_mask=None):
        # encoder output
        enc_output = self.encoder(enc_input, enc_mask)
        # decoder output
        dec_output = self.decoder(enc_output, enc_mask, dec_input, dec_mask=dec_mask)
        # fully connected layers output
        output = self.fcs(dec_output)        
        """if self.scale_prj:
            seq_logit *= self.d_model ** -0.5"""

        return output#.view(-1, seq_logit.size(2))