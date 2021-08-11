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
        self.position_enc = PositionalEncoding(150)
        self.dropout = nn.Dropout(p=cfg.DROPOUT_D)
        self.layer_norm = nn.LayerNorm(cfg.D_WORD_VEC, eps=1e-6)
        self.encoder_stack = nn.ModuleList([EncoderLayer(d_model=150, 
                                                        d_inner=4 * 150, 
                                                        n_head=8, 
                                                        d_k=32, 
                                                        d_v=32, 
                                                        dropout=cfg.DROPOUT_D)
                                                        for _ in range(6)])
        self.fc = nn.Linear(24, 1, bias=False)

    def forward(self, so3, text_vec, src_mask):
        # -- Forward
        """圖片上的normalize還有dropout在PositionwiseFeedForward"""
        if self.cfg.SCALE_EMB_D:
            input_vec = text_vec * self.cfg.D_MODEL_LIST_D[0] ** 0.5
        else:
            input_vec = text_vec
        input_vec = self.dropout(self.position_enc(input_vec))
        # 128 x 24 x 150
        input_vec  = self.layer_norm(input_vec)
        """# 128 x 24 x 150 直接用fasttext中的降為降成150
        enc_output, _ = self.encoder_text(input_vec, src_mask.unsqueeze(-2))"""
        # 128 x 24 x 300
        """TODO 這裡50是寫死的"""
        enc_output = torch.cat((input_vec, so3.repeat(1,1,50)), -1)
        for i in range(self.cfg.N_LAYERS_D):
            """
            enc_output: (batch, text_len, word_emb_len)
            src_mask: (batch, text_len) ==> unsqueeze(-2) ==> (batch, 1, text_len)
            """
            enc_output , _ = self.encoder_stack[i](enc_output , slf_attn_mask=src_mask.unsqueeze(-2))
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