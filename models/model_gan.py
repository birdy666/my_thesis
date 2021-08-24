import torch
import torch.nn as nn
import torch.nn.functional as F

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
                            
    net_g = Transformer(shareEncoder, decoder_g, cfg.FC_LIST_G).to(device)
    net_d = Transformer(shareEncoder, decoder_d, cfg.FC_LIST_D).to(device)
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

if __name__ == "__main__":
    class bb(nn.Module):
        def __init__(self, kk):
            super().__init__()
            self.a = nn.Linear(2,5)
            self.k = kk

    b = bb(nn.Linear(3,7))

    print(b)
    for t in b.k.parameters():
        t.requires_grad = False
    print(b.k)
    print(b.k.parameters())