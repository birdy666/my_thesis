import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from models.transformer import Encoder, Decoder, LinearDecoder
from models.sublayers import LinearWithChannel, MultiHeadAttention

def getModels(cfg, device, checkpoint=None):
    shareEncoder = Encoder(n_layers=cfg.ENC_PARAM_D.n_layers, 
                            d_model=72, 
                            d_inner_scale=cfg.ENC_PARAM_D.d_inner_scale, 
                            n_head=cfg.ENC_PARAM_D.n_head, 
                            d_k=cfg.ENC_PARAM_D.d_k, 
                            d_v=cfg.ENC_PARAM_D.d_v, 
                            dropout=cfg.ENC_PARAM_D.dropout, 
                            scale_emb=cfg.ENC_PARAM_D.scale_emb)
    decoder_g = LinearDecoder(72)    
    decoder_d = Decoder(n_layers=cfg.DEC_PARAM_D.n_layers, 
                            d_model=72, 
                            d_inner_scale=cfg.DEC_PARAM_D.d_inner_scale, 
                            n_head=cfg.DEC_PARAM_D.n_head, 
                            d_k=cfg.DEC_PARAM_D.d_k, 
                            d_v=cfg.DEC_PARAM_D.d_v, 
                            dropout=cfg.DEC_PARAM_D.dropout, 
                            scale_emb=cfg.DEC_PARAM_D.scale_emb)
                            
    net_g = Generator(decoder_g, device, cfg.D_WORD_VEC, cfg.NOISE_WEIGHT_G).to(device)
    net_d = Discriminator(shareEncoder, decoder_d, device, cfg.D_WORD_VEC, cfg.NOISE_WEIGHT_D).to(device)
    if checkpoint != None:
        print("Start from epoch " + str(cfg.START_FROM_EPOCH))        
        net_g.load_state_dict(checkpoint['model_g'])        
        net_d.load_state_dict(checkpoint['model_d'])
        print("Model loaded")
    else:
        #net_g.apply(init_weight)
        #net_d.apply(init_weight)
        print("Start a new training from epoch 0")
    return net_g, net_d

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight.data, 1.)

class Generator(nn.Module):
    def __init__(self, decoder, device, d_vec, noise_weight):
        super().__init__()
        #self.encoder = encoder
        self.decoder = decoder
        self.dropout = nn.Dropout(0.1)
        self.device = device
        self.d_vec = d_vec
        self.compress_size = d_vec 
        self.noise_weight = noise_weight
        """self.compress = nn.Sequential(
            nn.Linear(d_vec, self.compress_size),
            nn.LeakyReLU(0.2, inplace=True)
        )"""

    def forward(self, enc_output, input_mask, noise):
        #enc_output = self.encoder(input_text, input_mask)
        enc_output = self.dropout(enc_output)
        x = torch.ones_like(enc_output, dtype=torch.float32).to(self.device)  

        #dec_input = self.cross_attn(x, enc_output,enc_output, mask=input_mask.unsqueeze(-2))
        """enc_output_masked = enc_output.masked_fill(input_mask.unsqueeze(-1) == 0, 0)
        attn = torch.matmul(x , enc_output_masked.transpose(-2, -1))
        score = torch.matmul(attn, enc_output_masked)
        # each sentence has different length
        score = F.normalize(score, p=1, dim=-1)"""
        output = self.decoder(x,enc_output,mask=input_mask.unsqueeze(-2))
        return  output

class Discriminator(nn.Module):
    def __init__(self, encoder, decoder, device, d_vec, noise_weight):
        super().__init__()
        self.device = device
        self.d_vec = d_vec 
        self.compress_size = 72 
        self.noise_weight = noise_weight

        self.encoder = encoder
        self.decoder = decoder
        self.fcc = LinearWithChannel(self.compress_size,2,24)
         # compress text encoding first
        self.compress = nn.Sequential(
            nn.Linear(d_vec, self.compress_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, input_text, input_mask, rot_vec):
        batch_size = input_text.size(0)
        noise_tensor = torch.randn((batch_size, 24, 3), dtype=torch.float32).to(self.device)        
        rot_vec = rot_vec + self.noise_weight*noise_tensor
        # encoder output
        enc_output = self.encoder(enc_input=self.compress(input_text), enc_mask=input_mask)
        # decoder output
        dec_output = self.decoder(enc_output, 
                                enc_mask=input_mask, 
                                dec_input=F.pad(rot_vec,(0,self.compress_size-3), mode='constant', value=0.0),
                                #F.pad(rot_vec,(0,self.d_vec-3), mode='constant', value=0.0), 
                                dec_mask=torch.tensor(np.array([[1]+[1]*23]*batch_size)).to(self.device))
        
        output = self.fcc(dec_output)
        return output