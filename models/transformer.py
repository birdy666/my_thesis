import torch
import torch.nn as nn
import numpy as np
from models.layers import PositionalEncoding, EncoderLayer, DecoderLayer


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        """TODO 不了解"""
        self.position_enc = PositionalEncoding(150)
        self.dropout = nn.Dropout(p=cfg.DROPOUT_G)
        self.layer_norm = nn.LayerNorm(150, eps=1e-6)
        
        self.encoder_stack = nn.ModuleList([
                                            EncoderLayer(d_model=150, 
                                                        d_inner=4 * 150,
                                                        n_head=8, 
                                                        d_k=32, 
                                                        d_v=32, 
                                                        dropout=cfg.DROPOUT_G)
                                                        for _ in range(6)])

    def forward(self, input_vec, src_mask):
        # -- Forward
        """input_vec_noise = torch.cat((input_vec, noise), -1)"""        
        if self.cfg.SCALE_EMB_G:
            input_vec *= 150 ** 0.5
        """print("Before anything")
        print(input_vec[0])
        input_vec = self.dropout(self.position_enc(input_vec))
        print("After position_enc")
        print(input_vec[0])
        enc_output  = self.layer_norm(input_vec)
        print("After layernorm")
        print(enc_output[0])
        print("Diff")
        print(enc_output[0]-input_vec[0])
        print(sfsdf)"""
        enc_output = input_vec

        for enc_layer in self.encoder_stack:
            """
            enc_output: (batch, text_len, word_emb_len + noise_len)
            src_mask: (batch, text_len) ==> unsqueeze(-2) ==> (batch, 1, text_len)
            """
            enc_output = enc_layer(enc_output , slf_attn_mask=src_mask.unsqueeze(-2))  
        return enc_output

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        #self.position_enc = PositionalEncoding(cfg.D_WORD_VEC+cfg.NOISE_SIZE, n_position=cfg.N_POSITION_G)
        self.dropout = nn.Dropout(p=cfg.DROPOUT_G)
        self.layer_norm = nn.LayerNorm(150, eps=1e-6)
        
        self.decoder_stack = nn.ModuleList([
                                            DecoderLayer(d_model=150, 
                                                        d_inner=4 * 150,
                                                        n_head=8, 
                                                        d_k=32, 
                                                        d_v=32, 
                                                        dropout=cfg.DROPOUT_G)
                                                        for i in range(6)])

    def forward(self, dec_input, enc_output, enc_mask):
        # -- Forward
        """input_vec_noise = torch.cat((input_vec, noise), -1)"""        
        #if self.cfg.SCALE_EMB_G:
        #    input_vec *= self.cfg.D_MODEL_LIST_G[0] ** 0.5
        #input_vec = self.dropout(self.position_enc(input_vec))
        dec_input  = self.layer_norm(self.dropout(dec_input))

        for dec_layer in self.decoder_stack:
            """
            enc_output: (batch, text_len, word_emb_len + noise_len)
            src_mask: (batch, text_len) ==> unsqueeze(-2) ==> (batch, 1, text_len)
            """
            enc_output = dec_layer(dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=enc_mask.unsqueeze(-2)) 
        return enc_output


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.fcs = nn.Sequential(nn.Linear(150, 128, bias=False),
                                nn.Linear(128, 32, bias=False),
                                nn.Linear(32, 8, bias=False),
                                nn.Linear(8, 3, bias=False))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 



    def forward(self, input_vec, input_mask, dec_input):
        enc_output = self.encoder(input_vec, input_mask)
        dec_output = self.decoder(dec_input, enc_output, input_mask)
        output = self.fcs(dec_output)        
        """if self.scale_prj:
            seq_logit *= self.d_model ** -0.5"""

        return output#.view(-1, seq_logit.size(2))