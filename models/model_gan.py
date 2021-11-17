import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from models.transformer import Transformer, Encoder
from models.layers import EncoderLayer, DecoderLayer
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

class GG(nn.Module):
    def __init__(self, d_vec):
        super().__init__()
        self.fc1 = nn.Linear(24*d_vec//3, 24*d_vec//3)
        self.fc2 = nn.Linear(24*d_vec//3, 24*d_vec//3)
        self.fc3 = nn.Linear(24*d_vec//3, 24*d_vec//3)
        self.fc4 = nn.Linear(24*d_vec//3, 24*d_vec//3)
        self.fc5 = nn.Linear(24*d_vec//3, 4*d_vec)
        self.fc6 = nn.Linear(4*d_vec, 24*3)
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        self.dropout_3 = nn.Dropout(0.1)
        self.dropout_4 = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm1d(24*d_vec//3)
        self.bn2 = nn.BatchNorm1d(24*d_vec//3)
        self.bn3 = nn.BatchNorm1d(24*d_vec//3)
        self.bn4 = nn.BatchNorm1d(24*d_vec//3)
        self.act1 = nn.LeakyReLU(0.2)
        self.act2 = nn.LeakyReLU(0.2)
        self.act3 = nn.LeakyReLU(0.2)
        self.act4 = nn.LeakyReLU(0.2)
        

    def forward(self, x):
        x_res = x
        x = F.hardtanh(self.bn1(self.fc1(x)), min_val=-math.pi+0.0000000001, max_val=math.pi-0.0000000001)
        x = x_res + self.dropout_1(x)

        x_res = x
        x = F.hardtanh(self.bn2(self.fc2(x)), min_val=-math.pi+0.0000000001, max_val=math.pi-0.0000000001)     
        x = x_res + self.dropout_2(x)

        x_res = x
        x = F.hardtanh(self.bn3(self.fc3(x)), min_val=-math.pi+0.0000000001, max_val=math.pi-0.0000000001)        
        x = x_res + self.dropout_3(x)

        x_res = x
        x = F.hardtanh(self.bn4(self.fc4(x)), min_val=-math.pi+0.0000000001, max_val=math.pi-0.0000000001)        
        x = x_res + self.dropout_4(x)


        x = F.hardtanh(self.fc5(x), min_val=-math.pi+0.0000000001, max_val=math.pi-0.0000000001)           
        x = F.hardtanh(self.fc6(x), min_val=-math.pi+0.0000000001, max_val=math.pi-0.0000000001)        
        return x

# basically Encoder
class Generator(nn.Module):
    def __init__(self, enc_param, dec_param, fc_list, d_vec):
        super().__init__()
        self.encoder = Encoder(n_layers=enc_param.n_layers, 
                                d_model=enc_param.d_model, 
                                d_inner_scale=enc_param.d_inner_scale, 
                                n_head=enc_param.n_head, 
                                d_k=enc_param.d_k, 
                                d_v=enc_param.d_v, 
                                dropout=enc_param.dropout, 
                                scale_emb=enc_param.scale_emb)
        self.fc = LinearWithChannel(d_vec, d_vec//3, 24)
        self.dropout = nn.Dropout(0.1)
        self.d_vec = d_vec
        self.gg = GG(d_vec)

    def forward(self, input_text, input_mask, noise):
        batch_size = input_text.size(0)
        enc_output = self.encoder(input_text, input_mask)
        enc_output = self.dropout(self.fc(enc_output))
        x = torch.ones_like(enc_output, dtype=torch.float32).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))        
        x = x + +0.5*torch.randn((batch_size, 24, enc_output.size(-1)), dtype=torch.float32).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        attn = torch.matmul(x , enc_output.transpose(-2, -1))
        score = torch.matmul(attn, enc_output)        

        return  self.gg(score.view(batch_size, -1)).view(batch_size, 24, 3)

class Discriminator(nn.Module):
    def __init__(self, encoder, decoder, fc_list, d_vec):
        super().__init__()
        self.transformer = Transformer(encoder, decoder, fc_list)
        self.fc = LinearWithChannel(d_vec, d_vec//9, 24)
        self.fc2 = nn.Linear(24*d_vec//9, 1)
        #self.fc = nn.Linear(d_vec, 1, 24)
        self.dropout = nn.Dropout(0.05)
        self.d_vec = d_vec        
        
    def forward(self, input_text, input_mask, rot_vec):
        batch_size = input_text.size(0)
        noise_tensor = torch.randn((batch_size, 24, 3), dtype=torch.float32).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))        
        rot_vec = rot_vec + 0.25*noise_tensor
        output = self.transformer(enc_input=input_text, 
                                enc_mask=input_mask, 
                                dec_input=rot_vec.view(batch_size, 1, -1).repeat(1,24,1), 
                                dec_mask=torch.tensor(np.array([[1]+[1]*23]*batch_size)).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
        #output = self.dropout(self.fc(output.view(batch_size, -1)))
        output = self.fc(self.dropout(output)).view(batch_size, -1)
        output = self.fc2(output)
        output = (output*output)
        haha = F.relu((rot_vec*rot_vec).sum(-1)-(math.pi+0.01)**2)
        return output.mean()


if __name__ == "__main__":
    a = torch.tensor([0]*3)
    print()