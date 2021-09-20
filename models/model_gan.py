import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.transformer import Transformer
from models.layers import PositionalEncoding, EncoderLayer, DecoderLayer
from utils import get_noise_tensor



def getModels(cfg, device, checkpoint=None):
    net_g = Generator(cfg.ENC_PARAM_G, cfg.DEC_PARAM_G, cfg.FC_LIST_G).to(device)
    net_d = Discriminator(cfg.ENC_PARAM_D, cfg.DEC_PARAM_D, cfg.FC_LIST_D).to(device)
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
    def __init__(self, enc_param, dec_param, fc_list):
        super().__init__()
        self.transformer = Transformer(enc_param, dec_param, fc_list)

    def reg_output(self, output):
        norm = torch.norm(output, p=2, dim=-1, keepdim=False) # (batch, 24)
        # if the norm is smaller than pi than set it to pi, else dont change
        norm = torch.clamp(norm, 2*math.pi, math.inf) 
        output = torch.div(output, norm.unsqueeze(-1).repeat(1,1,3)) * (2*math.pi) # unsqueeze ==> (batch, 24, 1), repeat ==> (batch, 24, 3)
        return output

    def forward(self, input_text, input_mask, noise):
        output = self.transformer(enc_input=input_text, 
                                enc_mask=input_mask, 
                                dec_input=noise, 
                                dec_mask=None)        
        return self.reg_output(output)

class Discriminator(nn.Module):
    def __init__(self, enc_param, dec_param, fc_list):
        super().__init__()
        self.transformer = Transformer(enc_param, dec_param, fc_list)

    def reg_output(self, output):
        norm = torch.norm(output, p=2, dim=-1, keepdim=False) # (batch, 24)
        # if the norm is smaller than pi than set it to pi, else dont change
        norm = torch.clamp(norm, 2*math.pi, math.inf) 
        output = torch.div(output, norm.unsqueeze(-1).repeat(1,1,3)) * (2*math.pi) # unsqueeze ==> (batch, 24, 1), repeat ==> (batch, 24, 3)
        return output
        
    def forward(self, input_text, input_mask, rot_vec):
        output = self.transformer(enc_input=input_text, 
                                enc_mask=input_mask, 
                                dec_input=rot_vec.repeat(1,1,50), 
                                dec_mask=None)
        return self.reg_output(output)


if __name__ == "__main__":
    a = nn.ModuleList([nn.Dropout()  for i in range(5)])

    print(a[0])