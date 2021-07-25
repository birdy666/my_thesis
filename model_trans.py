import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import PositionalEncoding, EncoderLayer
from utils import get_noise_tensor



def getModels(cfg, device, checkpoint=None):
    net_g = Generator(cfg).to(device)
    net_d = Discriminator(cfg).to(device)
    if checkpoint != None:
        print("Start from epoch " + str(cfg.START_FROM_EPOCH))        
        net_g.load_state_dict(checkpoint['model_g'])        
        net_g.load_state_dict(checkpoint['model_d'])
        print("Model loaded")
    else:
        net_g.apply(init_weight)
        net_d.apply(init_weight)
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
        self.position_enc = PositionalEncoding(cfg.D_WORD_VEC+cfg.NOISE_SIZE, n_position=cfg.N_POSITION_G)
        self.dropout = nn.Dropout(p=cfg.DROPOUT_G)
        self.layer_norm = nn.LayerNorm(cfg.D_MODEL_LIST_G[0], eps=1e-6)
        self.encoder_stack = nn.ModuleList([
                                            EncoderLayer(d_model=cfg.D_MODEL_LIST_G[i], 
                                                        d_inner=cfg.D_MODEL_LIST_G[i] * cfg.D_INNER_SCALE_G, 
                                                        d_out = cfg.D_MODEL_LIST_G[i+1],
                                                        n_head=cfg.N_HEAD_LIST_G[i], 
                                                        d_k=cfg.D_K_LIST_G[i], 
                                                        d_v=cfg.D_V_LIST_G[i], 
                                                        dropout=cfg.DROPOUT_G)
                                                        for i in range(cfg.N_LAYERS_G)])
         
    def forward(self, noise, input_vec, src_mask, return_attns=False):
        # -- Forward
        input_vec_noise = torch.cat((input_vec, noise), -1)
        """這裡應該不用每一層都做， 圖片上的normalize還有dropout在PositionwiseFeedForward"""
        if self.cfg.SCALE_EMB_G:
            input_vec_noise *= self.cfg.D_MODEL_LIST_G[0] ** 0.5
        input_vec_noise = self.dropout(self.position_enc(input_vec_noise))
        enc_output  = self.layer_norm(input_vec_noise)

        for enc_layer in self.encoder_stack:
            """
            enc_output: (batch, text_len, word_emb_len)
            src_mask: (batch, text_len) ==> unsqueeze(-2) ==> (batch, 1, text_len)
            """
            enc_output , _ = enc_layer(enc_output , slf_attn_mask=src_mask.unsqueeze(-2))
           
        return enc_output

class Discriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.position_enc = PositionalEncoding(cfg.D_WORD_VEC, n_position=cfg.N_POSITION_D)
        self.dropout = nn.Dropout(p=cfg.DROPOUT_D)
        self.layer_norm = nn.LayerNorm(cfg.D_MODEL_LIST_D[0], eps=1e-6)
        self.encoder_stack = nn.ModuleList([
                                            EncoderLayer(d_model=cfg.D_MODEL_LIST_D[i], 
                                                        d_inner=cfg.D_MODEL_LIST_D[i] * cfg.D_INNER_SCALE_D, 
                                                        d_out = cfg.D_MODEL_LIST_D[i+1],
                                                        n_head=cfg.N_HEAD_LIST_D[i], 
                                                        d_k=cfg.D_K_LIST_D[i], 
                                                        d_v=cfg.D_V_LIST_D[i], 
                                                        dropout=cfg.DROPOUT_D)
                                                        for i in range(cfg.N_LAYERS_D)])
        self.encoder_last = EncoderLayer(d_model=6, 
                                        d_inner=24, 
                                        d_out =1,
                                        n_head=2, 
                                        d_k=3, 
                                        d_v=3, 
                                        dropout=cfg.DROPOUT_D)

        self.fc = nn.Linear(24, 1, bias=False)

    def forward(self, so3, text_vec, src_mask):
        # -- Forward
        input_vec = text_vec.clone()
        """這裡應該不用每一層都做， 圖片上的normalize還有dropout在PositionwiseFeedForward"""
        if self.cfg.SCALE_EMB_D:
            input_vec *= self.cfg.D_MODEL_LIST_D[0] ** 0.5
        input_vec = self.dropout(self.position_enc(input_vec))
        enc_output  = self.layer_norm(input_vec)

        for enc_layer in self.encoder_stack:
            """
            enc_output: (batch, text_len, word_emb_len)
            src_mask: (batch, text_len) ==> unsqueeze(-2) ==> (batch, 1, text_len)
            """
            enc_output , _ = enc_layer(enc_output , slf_attn_mask=src_mask.unsqueeze(-2))
        
        # (128, 24, 3+3)
        so3_enc_output = torch.cat((so3, enc_output), -1)
        output, _ =  self.encoder_last(so3_enc_output, slf_attn_mask=None)
        output = self.fc(output.view(-1, self.cfg.JOINT_NUM))
        #output = F.sigmoid(output)
        return output


if __name__ == "__main__":
    a = [1,2,3,4,5]
    for i in range(4, 0-1, -1):
        #print(i)
        print(a[i])