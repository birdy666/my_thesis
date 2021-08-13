import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import Transformer
from models.layers import PositionalEncoding, EncoderLayer
from utils import get_noise_tensor



def getModels(cfg, device, checkpoint=None):
    net_g = Generator(cfg).to(device)
    net_d = Discriminator(cfg).to(device)
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
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = Transformer(cfg)

    def forward(self, noise, input_vec, input_mask):
        output = self.transformer(input_vec, input_mask, noise)
        return output

class Discriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.position_enc = PositionalEncoding(cfg.D_WORD_VEC)
        self.dropout = nn.Dropout(p=cfg.DROPOUT_D)
        self.layer_norm = nn.LayerNorm(cfg.D_WORD_VEC, eps=1e-6)
        self.encoder_stack = nn.ModuleList([EncoderLayer(d_model=cfg.D_MODEL_LIST_D[i], 
                                                        d_inner=cfg.D_MODEL_LIST_D[i] * cfg.D_INNER_SCALE_D, 
                                                        n_head=cfg.N_HEAD_LIST_D[i], 
                                                        d_k=cfg.D_K_LIST_D[i], 
                                                        d_v=cfg.D_V_LIST_D[i], 
                                                        dropout=cfg.DROPOUT_D)
                                                        for i in range(cfg.N_LAYERS_D)])  
        self.fc_stack = nn.ModuleList([nn.Linear(cfg.D_MODEL_LIST_D[i], cfg.D_MODEL_LIST_D[i+1], bias=False) for i in range(cfg.N_LAYERS_D)])  
        self.fc = nn.Linear(24, 1, bias=False)

    def forward(self, so3, sentence_vec, src_mask=None):
        # -- Forward
        """圖片上的normalize還有dropout在PositionwiseFeedForward"""
        if self.cfg.SCALE_EMB_D:
            sentence_vec = sentence_vec * self.cfg.D_MODEL_LIST_D[0] ** 0.5
        else:
            sentence_vec = sentence_vec
        sentence_vec = self.dropout(self.position_enc(sentence_vec))
        # 128 x 24 x 150
        sentence_vec  = self.layer_norm(sentence_vec)
        
        
        """# 128 x 24 x 150 直接用fasttext中的降為降成150
        enc_output, _ = self.encoder_text(input_vec, src_mask.unsqueeze(-2))"""
        # 128 x 24 x 300
        """TODO 這裡50是寫死的"""
        enc_output = torch.cat((sentence_vec, so3.repeat(1,1,50)), -1)
        for i in range(self.cfg.N_LAYERS_D):
            """
            enc_output: (batch, text_len, word_emb_len)
            src_mask: (batch, text_len) ==> unsqueeze(-2) ==> (batch, 1, text_len)
            """
            enc_output = self.encoder_stack[i](enc_output)
            enc_output = self.fc_stack[i](enc_output)
        """
        # (128, 24, 3+3)
        so3_enc_output = torch.cat((so3, enc_output), -1)
        output, _ =  self.encoder_last(so3_enc_output, slf_attn_mask=None)"""
        # 128 x 24 x 1
        output = self.fc(enc_output.view(-1, self.cfg.JOINT_NUM))
        #output = F.sigmoid(output)
        return output


if __name__ == "__main__":
    a = nn.ModuleList([nn.Dropout()  for i in range(5)])

    print(a[0])