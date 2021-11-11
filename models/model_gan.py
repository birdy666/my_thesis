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

class LinearWithChannel(nn.Module):
    # https://github.com/pytorch/pytorch/issues/36591
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()
        
        #initialize weights
        self.w = torch.nn.Parameter(torch.zeros(channel_size, input_size, output_size))
        self.b = torch.nn.Parameter(torch.zeros(1, channel_size, output_size))
        
        #change weights to kaiming
        self.reset_parameters(self.w, self.b)
        
    def reset_parameters(self, weights, bias):
        
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
    
    def forward(self, x):
        b, c, i = x.size()
        return ( x.unsqueeze(-2) @ self.w).squeeze(-2) + self.b

# basically Encoder
class Generator(nn.Module):
    def __init__(self, enc_param, dec_param, fc_list, d_vec):
        super().__init__()
        self.transformer = Transformer(enc_param, dec_param, fc_list)
        self.fc = LinearWithChannel(d_vec, 4, 24)
        self.dropout = nn.Dropout(0.05)
        self.conv=nn.Conv1d(in_channels=24*4,out_channels=24*4,kernel_size=d_vec//4,groups=24*4)
        self.d_vec = d_vec

    def forward(self, input_text, input_mask, noise):
        batch_size = input_text.size(0)
        output = self.transformer(enc_input=input_text, 
                                enc_mask=input_mask, 
                                dec_input=noise, 
                                dec_mask=torch.tensor(np.array([[1]+[1]*23]*batch_size)).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
                                #torch.tensor(np.array([[1]+[0]*23]*batch_size)).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        #output = F.hardtanh(self.fc(output).view(batch_size, 24, 24, 3), min_val=-math.pi, max_val=math.pi)
        #output = F.hardtanh(self.dropout(self.fc(output.view(batch_size, -1))).view(batch_size,24,3), min_val=-math.pi, max_val=math.pi)
        
        #output = self.fc(output.view(batch_size, -1)).view(batch_size,24,4)
        #output = self.conv(output.view(batch_size,24*4,self.d_vec//4)).view(batch_size,24,4)
        output = self.dropout(self.fc(output))
        return  F.hardtanh(output, min_val=-math.pi+0.0000000001, max_val=math.pi-0.0000000001) 
class Discriminator(nn.Module):
    def __init__(self, encoder, decoder, fc_list, d_vec):
        super().__init__()
        self.transformer = Transformer(encoder, decoder, fc_list)
        self.fc = nn.Linear(24*d_vec, 24*1, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.d_vec = d_vec        
        
    def forward(self, input_text, input_mask, rot_vec):
        batch_size = input_text.size(0)
        output = self.transformer(enc_input=input_text, 
                                enc_mask=input_mask, 
                                dec_input=rot_vec.view(batch_size, 1, 24*4).repeat(1,24,1), 
                                dec_mask=torch.tensor(np.array([[1]+[0]*23]*batch_size)).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
        #output = self.dropout(self.fc(output.view(batch_size, -1)))
        return output[:,:1,:].sum(-1)


if __name__ == "__main__":
    a = torch.tensor([0]*3)
    print()