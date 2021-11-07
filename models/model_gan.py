import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from models.transformer import Transformer, Encoder
from models.layers import PositionalEncoding, EncoderLayer, DecoderLayer
from utils import get_noise_tensor
import numpy as np



def getModels(cfg, device, checkpoint=None):  
    net_g = Generator(cfg.ENC_PARAM_G, cfg.DEC_PARAM_G, cfg.FC_LIST_G, cfg.D_WORD_VEC).to(device)
    net_d = Discriminator(cfg.ENC_PARAM_D, cfg.DEC_PARAM_D, cfg.FC_LIST_D, cfg.D_WORD_VEC).to(device)

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
    def __init__(self, enc_param, dec_param, fc_list, d_vec):
        super().__init__()
        self.transformer = Transformer(enc_param, dec_param, fc_list)
        self.fc = nn.Linear(24*d_vec, 24*3, bias=False)

    def forward(self, input_text, input_mask, noise):
        batch_size = input_text.size(0)
        output = self.transformer(enc_input=input_text, 
                                enc_mask=input_mask, 
                                dec_input=noise, 
                                dec_mask=torch.tensor(np.array([[1]+[1]*23]*batch_size)).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
                                #torch.tensor(np.array([[1]+[0]*23]*batch_size)).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        #output = F.hardtanh(self.fc(output).view(batch_size, 24, 24, 3), min_val=-math.pi, max_val=math.pi)
        output = F.hardtanh(self.fc(output.view(batch_size, -1)).view(batch_size,24,3), min_val=-math.pi, max_val=math.pi)  
        #return output[:,:1,:,:].squeeze(1)
        return output

class Discriminator(nn.Module):
    def __init__(self, encoder, decoder, fc_list, d_vec):
        super().__init__()
        self.transformer = Transformer(encoder, decoder, fc_list)
        self.fc = nn.Linear(24*d_vec, 24*1, bias=False)
        self.d_vec = d_vec
        
    def forward(self, input_text, input_mask, rot_vec):
        batch_size = input_text.size(0)
        output = self.transformer(enc_input=input_text, 
                                enc_mask=input_mask, 
                                dec_input=rot_vec.repeat(1,1,self.d_vec//3), 
                                dec_mask=None)
        output = self.fc(output.view(batch_size, -1))
        return output


if __name__ == "__main__":
    a = torch.tensor([0]*3)
    print()