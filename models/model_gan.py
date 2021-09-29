import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from models.transformer import Transformer, Encoder, Decoder
from models.layers import PositionalEncoding, EncoderLayer, DecoderLayer
from utils import get_noise_tensor



def getModels(cfg, device, checkpoint=None):  
    shareEncoder = Encoder(n_layers=cfg.ENC_PARAM.n_layers, 
                            d_model=cfg.ENC_PARAM.d_model, 
                            d_inner_scale=cfg.ENC_PARAM.d_inner_scale, 
                            n_head=cfg.ENC_PARAM.n_head, 
                            d_k=cfg.ENC_PARAM.d_k, 
                            d_v=cfg.ENC_PARAM.d_v, 
                            dropout=cfg.ENC_PARAM.dropout, 
                            scale_emb=cfg.ENC_PARAM.scale_emb)

    decoder_g = Decoder(n_layers=cfg.DEC_PARAM_G.n_layers, 
                            d_model=cfg.DEC_PARAM_G.d_model, 
                            d_inner_scale=cfg.DEC_PARAM_G.d_inner_scale, 
                            n_head=cfg.DEC_PARAM_G.n_head, 
                            d_k=cfg.DEC_PARAM_G.d_k, 
                            d_v=cfg.DEC_PARAM_G.d_v, 
                            dropout=cfg.DEC_PARAM_G.dropout, 
                            scale_emb=cfg.DEC_PARAM_G.scale_emb)

    decoder_d = Decoder(n_layers=cfg.DEC_PARAM_D.n_layers, 
                            d_model=cfg.DEC_PARAM_D.d_model, 
                            d_inner_scale=cfg.DEC_PARAM_D.d_inner_scale, 
                            n_head=cfg.DEC_PARAM_D.n_head, 
                            d_k=cfg.DEC_PARAM_D.d_k, 
                            d_v=cfg.DEC_PARAM_D.d_v, 
                            dropout=cfg.DEC_PARAM_D.dropout, 
                            scale_emb=cfg.DEC_PARAM_D.scale_emb)

    net_g = Generator(shareEncoder, decoder_g, cfg.FC_LIST_G).to(device)
    net_d = Discriminator(shareEncoder, decoder_d, cfg.FC_LIST_D).to(device)
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


# basically Encoder
class Generator(nn.Module):
    def __init__(self, encoder, decoder, fc_list):
        super().__init__()
        self.encoder = encoder
        self.transformer = Transformer(self.encoder, decoder, fc_list)

        layers = []
        for i in range(6):
            layers.append(nn.Linear(150, 150))
        self.fcs = nn.Sequential(*layers)

    def reg_output(self, output):
        norm = torch.norm(output, p=2, dim=-1, keepdim=False) # (batch, 24)
        # if the norm is smaller than pi than set it to pi, else dont change
        norm = torch.clamp(norm, 2*math.pi, math.inf) 
        output = torch.div(output, norm.unsqueeze(-1).repeat(1,1,3)) * (2*math.pi) # unsqueeze ==> (batch, 24, 1), repeat ==> (batch, 24, 3)
        return output

    def forward(self, input_text, input_mask, noise):
        noise = self.fcs(noise)
        output = self.transformer(enc_input=input_text, 
                                enc_mask=input_mask, 
                                dec_input=noise, 
                                dec_mask=torch.tensor(np.array([[0]*24]*512)).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))        
        return self.reg_output(output)

class Discriminator(nn.Module):
    def __init__(self, encoder, decoder, fc_list):
        super().__init__()
        self.transformer = Transformer(encoder, decoder, fc_list)
        
    def forward(self, input_text, input_mask, rot_vec):
        output = self.transformer(enc_input=input_text, 
                                enc_mask=input_mask, 
                                dec_input=rot_vec.repeat(1,1,50), 
                                dec_mask=None)
        return output


if __name__ == "__main__":
    a = torch.tensor([0]*3)
    print()