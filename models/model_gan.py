import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from models.transformer import Decoder_G, Decoder_D
from models.sublayers import LinearWithChannel, LinearWithChannel_2, PositionalEncoder

def getModels(cfg, device, checkpoint=None):
    decoder_g = Decoder_G(n_layers=cfg.DEC_PARAM_G.n_layers, 
                            d_model=cfg.DEC_PARAM_G.d_model, 
                            d_inner_scale=cfg.DEC_PARAM_G.d_inner_scale, 
                            n_head=cfg.DEC_PARAM_G.n_head, 
                            d_k=cfg.DEC_PARAM_G.d_k, 
                            d_v=cfg.DEC_PARAM_G.d_v, 
                            dropout=cfg.DEC_PARAM_G.dropout, 
                            scale_emb=cfg.DEC_PARAM_G.scale_emb)    
    decoder_d = Decoder_D(n_layers=cfg.DEC_PARAM_D.n_layers, 
                            d_model=cfg.DEC_PARAM_D.d_model, 
                            d_inner_scale=cfg.DEC_PARAM_D.d_inner_scale, 
                            n_head=cfg.DEC_PARAM_D.n_head, 
                            d_k=cfg.DEC_PARAM_D.d_k, 
                            d_v=cfg.DEC_PARAM_D.d_v, 
                            dropout=cfg.DEC_PARAM_D.dropout, 
                            scale_emb=cfg.DEC_PARAM_D.scale_emb)
                            
    net_g = Generator(decoder_g, cfg.D_WORD_VEC).to(device)
    net_d = Discriminator(decoder_d, cfg.D_WORD_VEC, cfg.NOISE_WEIGHT_D, device).to(device)
    if checkpoint != None:
        print("Start from epoch " + str(cfg.START_FROM_EPOCH))        
        net_g.load_state_dict(checkpoint['model_g'])        
        net_d.load_state_dict(checkpoint['model_d'])
        print("Model loaded")
    else:
        print("Start a new training from epoch 0")
    return net_g, net_d

class Generator(nn.Module):
    def __init__(self, decoder, d_vec):
        super().__init__()
        self.embedding = nn.Embedding(27297, d_vec)
        self.decoder = decoder
        self.d_vec = d_vec
        #self.fc0 = LinearWithChannel_2(d_vec,d_vec,24)
        self.fc = nn.Linear(d_vec, 3)      
        self.pe = PositionalEncoder(d_vec)        
        self.dropout = nn.Dropout(p=0.1)  
    
    def forward(self, captions_index, input_mask, noise):
        captions_emb = self.embedding(captions_index)
        captions_emb = self.pe(captions_emb)
        enc_output = captions_emb
        output, attn = self.decoder(noise,enc_output,enc_mask=input_mask)
        output = self.fc(output)
        return F.hardtanh(output, min_val=-math.pi, max_val=math.pi), attn

class Discriminator(nn.Module):
    def __init__(self, decoder, d_vec, noise_weight, device):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(27297, d_vec)
        self.decoder = decoder
        self.noise_weight = noise_weight
        
        self.fc1 = nn.Linear(24*3, d_vec)
        self.fc2 = nn.Linear(d_vec, 1)
        self.layer_norm = nn.LayerNorm(d_vec, eps=1e-6)
        self.pe = PositionalEncoder(d_hid=d_vec) 
        self.d_vec=d_vec
        
    def forward(self, captions_emb, input_mask, rot_vec):
        # do embedding outside because of gradient penalty
        captions_emb = self.pe(captions_emb)
        
        enc_output = captions_emb
        noise_tensor = torch.randn((captions_emb.size(0), 24, 3), dtype=torch.float32).to(self.device)        
        
        #rot_vec = rot_vec* (self.d_vec ** 0.5)
        #rot_vec = self.pe_2(rot_vec)
        
        rot_vec = rot_vec + self.noise_weight*noise_tensor 
        rot_vec = self.fc1(rot_vec.view(-1,1,24*3))

        output, attn = self.decoder(rot_vec, enc_output, enc_mask=input_mask)
        output = self.fc2(output)
        return F.hardtanh(output, min_val=-50, max_val=50), attn