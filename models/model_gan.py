import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from models.transformer import Encoder, Decoder, LinearDecoder
from models.sublayers import LinearWithChannel, MultiHeadAttention

def getModels(cfg, device, checkpoint=None):
    encoder_g = Encoder(n_layers=cfg.ENC_PARAM_G.n_layers, 
                            d_model=cfg.ENC_PARAM_G.d_model, 
                            d_inner_scale=cfg.ENC_PARAM_G.d_inner_scale, 
                            n_head=cfg.ENC_PARAM_G.n_head, 
                            d_k=cfg.ENC_PARAM_G.d_k, 
                            d_v=cfg.ENC_PARAM_G.d_v, 
                            dropout=cfg.ENC_PARAM_G.dropout, 
                            scale_emb=cfg.ENC_PARAM_G.scale_emb)
    encoder_d = Encoder(n_layers=cfg.ENC_PARAM_D.n_layers, 
                            d_model=cfg.ENC_PARAM_D.d_model, 
                            d_inner_scale=cfg.ENC_PARAM_D.d_inner_scale, 
                            n_head=cfg.ENC_PARAM_D.n_head, 
                            d_k=cfg.ENC_PARAM_D.d_k, 
                            d_v=cfg.ENC_PARAM_D.d_v, 
                            dropout=cfg.ENC_PARAM_D.dropout, 
                            scale_emb=cfg.ENC_PARAM_D.scale_emb)
    decoder_g = LinearDecoder(cfg.ENC_PARAM_G.d_model)    
    decoder_d = Decoder(n_layers=cfg.DEC_PARAM_D.n_layers, 
                            d_model=cfg.DEC_PARAM_D.d_model, 
                            d_inner_scale=cfg.DEC_PARAM_D.d_inner_scale, 
                            n_head=cfg.DEC_PARAM_D.n_head, 
                            d_k=cfg.DEC_PARAM_D.d_k, 
                            d_v=cfg.DEC_PARAM_D.d_v, 
                            dropout=cfg.DEC_PARAM_D.dropout, 
                            scale_emb=cfg.DEC_PARAM_D.scale_emb)
                            
    net_g = Generator(encoder_g, decoder_g, device, cfg.D_WORD_VEC, cfg.NOISE_WEIGHT_G).to(device)
    net_d = Discriminator(encoder_d, decoder_d, device, cfg.D_WORD_VEC, cfg.NOISE_WEIGHT_D).to(device)
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
    def __init__(self, encoder, decoder, device, d_vec, noise_weight):
        super().__init__()
        self.embedding = nn.Embedding(27297, d_vec)
        self.dropout_1 = nn.Dropout(0.1)
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.d_vec = d_vec
        self.noise_weight = noise_weight

    def forward(self, captions, input_mask, noise):
        emb = self.dropout_1(self.embedding(captions))
        enc_output = self.encoder(enc_input=emb, enc_mask=input_mask)
        x = torch.ones_like(enc_output, dtype=torch.float32).to(self.device)       
        output = self.decoder(x,enc_output,mask=input_mask.unsqueeze(-2))
        return  output

class Discriminator(nn.Module):
    def __init__(self, encoder, decoder, device, d_vec, noise_weight):
        super().__init__()
        self.device = device
        self.d_vec = d_vec 
        self.noise_weight = noise_weight
        self.embedding = nn.Embedding(27297, d_vec)
        self.dropout_1 = nn.Dropout(0.1)

        self.encoder = encoder
        self.decoder = decoder
        self.fcc = LinearWithChannel(self.d_vec,2,24)
        
    def forward(self, captions, input_mask, rot_vec):
        batch_size = captions.size(0)
        noise_tensor = torch.randn((batch_size, 24, 3), dtype=torch.float32).to(self.device)        
        rot_vec = rot_vec + self.noise_weight*noise_tensor

        emb = self.dropout_1(self.embedding(captions))
        # encoder output        
        enc_output = self.encoder(enc_input=emb, enc_mask=input_mask)
        # decoder output
        dec_output = self.decoder(enc_output, 
                                enc_mask=input_mask, 
                                dec_input=F.pad(rot_vec,(0,self.d_vec-3), mode='constant', value=0.0),
                                #F.pad(rot_vec,(0,self.d_vec-3), mode='constant', value=0.0), 
                                dec_mask=torch.tensor(np.array([[1]+[1]*23]*batch_size)).to(self.device))
        
        output = self.fcc(dec_output)
        return output