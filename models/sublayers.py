import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from torch.autograd import Variable

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        # square root of d_k 
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q , k.transpose(-2, -1)) / self.temperature
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        #attn = self.dropout(attn)
        output = torch.matmul(attn, v)        
        return output

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)        

    # q = k = v = enc_input
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        score = self.attention(q, k, v, mask=mask)

        score_concat = score.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        output = self.fc(score_concat)
        output = self.dropout(output) 
        output += residual
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, d_hid=128, n_position=24, r=False):
        super(PositionalEncoder, self).__init__()
        self.r = r
        if r:
            self.register_buffer('pos_table', self._get_table(n_position))
        # Not a parameter
        else:
            self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
    def _get_table(self, n_position):
        table = np.array([i for i in range(n_position)]) / n_position
        return torch.FloatTensor(table).unsqueeze(0).unsqueeze(-1)

    def _get_table_r(self, n_position):
        table = np.array([0.1*i for i in range(n_position)]) / n_position
        return torch.FloatTensor(table).unsqueeze(0).unsqueeze(-1)


    def forward(self, x):
        if self.r:
            return x * self.pos_table.clone().detach()
        else:
            return x + self.pos_table.clone().detach()

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
        return ( x.unsqueeze(-2) @ self.w).squeeze(-2) + self.b



class LinearWithChannel_2(nn.Module):
    # https://github.com/pytorch/pytorch/issues/36591
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel_2, self).__init__()        
        #initialize weights
        self.w = torch.nn.Parameter(torch.zeros(channel_size, input_size, output_size))
        self.b = torch.nn.Parameter(torch.zeros(channel_size, 1, output_size))        
        #change weights to kaiming
        self.reset_parameters(self.w, self.b)
        
    def reset_parameters(self, weights, bias):        
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
    
    def forward(self, x):
        return ( x.unsqueeze(-2) @ self.w  + self.b).squeeze(-2)