
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os

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
        # matmul會把Q、K後兩維進行矩陣乘法 timex64 , 64xtime ==> time x time
        """
        q, k, v: (batch, head, text_len, d_k)
        attn: (batch, head, text_len, text_len)
        """
        attn = torch.matmul(q , k.transpose(-2, -1)) / self.temperature

        if mask is not None:
            # 把mask原本是0的轉變成一個小的數-1e9，這樣經過softmax之後概率接近0但不為0
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        # 大致流程 (Q, K) ==> matmul ==> scale ==> mask(if any) ==> softmax ==> 和V做matmul
        """
        output: (batch, head, text_len , d_k)
        attn: (batch, head, text_len, text_len)
        """
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
        # 他基本上把8個head的不同矩陣黏一起所以forward時才會再用view把512拆成 8, 64
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        # 補充講義上的W0 用來把全部黏再一起的z0~zn變回d_model長， 但這裡要注意的是我的d_model != n_head * d_k
        # 為啥不在這裡直接把惟度轉換成d_out是因為下面還有一個residual,如果in/out不同維度很麻煩
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    
    """
    q = k = v = enc_input
    """
    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # q,k,v: (batch, text_len, d_model) 
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # 從這裡可以看出，d_model=head*d_k 應該不是必然 而是他在定義nn.Linear時剛好設成一樣
        """
        在乘上w_qs, w_ks, w_vs之前 qkv都是等於 enc_input, 講義上 q = W*a etc
        q, k, v: (batch, text_len, d_model) ==> w_qs ==> (batch, text_len, head*d_k ) 
                    ==> view ==> (batch, text_len, head, d_k)
        """
        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)

        # Transpose 變(batch, head, text_len, d_k) 這是attention要求的形狀
        """
        q, k, v: (batch, text_len, head, d_k) ==> transpose(1, 2) ==> (batch, head, text_len, d_k)
        """ 
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            """
            mask: (batch, 1, text_len) ==> unsqueeze(1) ==> (batch, 1, 1, text_len)
            """
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        """
        q: (batch, head, text_len , d_k)
        attn: (batch, head, text_len, text_len)
        """
        q, attn = self.attention(q, k, v, mask=mask)

        # 把多頭產生的結果黏在一起 8 個頭 dv = 64 , 8x64=512. 
        """
        q: (batch, head, text_len , d_k) ==> (batch, text_len, head*d_k)
        """
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        """
        q: (batch, text_len, head*d_k) ==> (batch, text_len, d_model)
        """
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)
        """
        q: (batch, text_len, d_model)
        attn: (batch, head, text_len, text_len)
        """
        return q, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
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

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(max_len, d_model))

    def _get_sinusoid_encoding_table(self, max_len, d_model):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(max_len)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
    def forward(self, enc_input, slf_attn_mask=None):
        # 這裡的output是多頭產生的結果，並黏在一起
        """
        enc_input:  (batch, text_len, d_model)
        slf_attn_mask:  (batch, 1, text_len)
        """
        # enc_input & enc_output have the same shape in the paper, 
        """
        enc_output: (batch, text_len, d_model)
        enc_slf_attn: (batch, head, text_len, text_len)
        """
        enc_output, _ = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output

class DecoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_dec_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):        
        dec_output, _ = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        enc_dec_output, _ = self.enc_dec_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        enc_dec_output = self.pos_ffn(enc_dec_output)

        return enc_output

if __name__ == "__main__":
    x = torch.randn(1, 20, 30)  # 输入的维度是（128，20）

    m = torch.nn.Linear(30, 15)  # 20,30是指维度

    output = m(x)


    print('output.shape:\n', output.shape)