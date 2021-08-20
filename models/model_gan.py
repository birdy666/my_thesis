import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, input_text, input_mask, noise):
        output = self.transformer(enc_input=input_text, 
                                enc_mask=input_mask, 
                                dec_input=noise, 
                                dec_mask=None)
        return output

class Discriminator(nn.Module):
    def __init__(self, enc_param, dec_param, fc_list):
        super().__init__()
        self.transformer = Transformer(enc_param, dec_param, fc_list)

    def forward(self, input_text, input_mask, rot_vec):
        output = self.transformer(enc_input=input_text, 
                                enc_mask=input_mask, 
                                dec_input=rot_vec*0.1, 
                                dec_mask=None)
        return output


if __name__ == "__main__":
    a = nn.ModuleList([nn.Dropout()  for i in range(5)])

    print(a[0])