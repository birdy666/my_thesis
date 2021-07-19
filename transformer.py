
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

"""d_word_vec=512, 
d_model=512, 
d_inner=2048,
n_layers=6, 
n_head=8, 
d_k=64, 
d_v=64, 
dropout=0.1, 
n_position=200

Word embedding 把單字轉成512維的向量 a
q = a*W_q    a是1x512, W_q是512x64 所以q是1x64
k同理
"""

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        # square root of d_k 
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # temperature here is used to scale down the matmul of Q,K  to counter exploding gradients.
        # Q, K 都是四維矩陣，(batch, head, time, 64)，其中head=8,time是句子長度
        # matmul會把Q、K後兩維進行矩陣乘法
        attn = torch.matmul(q , k.transpose(-2, -1)) / self.temperature

        if mask is not None:
            # 把mask原本是0的轉變成一個小的數-1e9，這樣經過softmax之後概率接近0但不為0
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        # 大致流程 (Q, K) ==> matmul ==> scale ==> mask(if any) ==> softmax ==> 和V做matmul
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        # dk is the dimension of queries (Q) and keys(K)
        self.d_k = d_k
        # 照理說V德對不是應該和QK一樣嗎
        self.d_v = d_v

        # 多頭的attention基本上就是有n個q, k
        # nn.Linear(512, 8 * 64, bias=False) ==> 512x512的矩陣
        # 他基本上把8個head地個不同矩陣黏一起所以forward時才會再用view把512拆成 8, 64
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    """這裡我有點不懂，明明q=k=v= 單字經過embedding後的512維向量(EncoderLayer forward中的enc_input)
    要經過linear後才會變成q,k,v怎麼這裡就叫他們q,k,v? 而且既然一樣幹嘛要傳三個...    
    """
    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # sz_b = batch size
        # q,k,v 的形狀是 (batch, time, 512) 
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # 經過linear後變成 (batch, time, 512)， 透過view變成 (batch, time, 8, 64)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose 變(batch, 8, time, 64) 這是attention要求的形狀
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # 把多頭產生的結果黏在一起 8 個頭 dv = 64 , 8x64=512. 
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in=512, d_hid=2048, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        # 這裡的output是多頭產生的結果，並黏在一起
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
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

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


if __name__ == "__main__":
    a = np.array([1,1,1,1,2,2,2,2,3,3,3,3])
    print(a.reshape(3,4))