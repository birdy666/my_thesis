import torch
import torch.nn as nn

from transformer import PositionalEncoding, EncoderLayer

def getModel(cfg):
    return Generator(cfg), Discriminator(cfg)

# basically Encoder
class Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.position_enc = PositionalEncoding(cfg.D_WORD_VEC, n_position=cfg.N_POSITION)
        self.dropout = nn.Dropout(p=cfg.DROPOUT)
        self.layer_norm = nn.LayerNorm(cfg.D_MODEL_LIST[0], eps=1e-6)
        self.encoder_stack = nn.ModuleList([
                                            EncoderLayer(d_model=cfg.D_MODEL_LIST[i], 
                                                        d_inner=cfg.D_MODEL_LIST[i] * cfg.D_INNER_SCALE, 
                                                        n_head=cfg.N_HEAD_LIST[i], 
                                                        d_k=cfg.D_K_LIST[i], 
                                                        d_v=cfg.D_V_LIST[i], 
                                                        dropout=cfg.DROPOUT)
                                                        for i in range(cfg.N_LAYERS)])
        
        
    
    """不知道mask要傳甚麼嗚嗚 TODO"""
    def forward(self, noise, input_vec, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        input_vec = input_vec.copy()
        """這裡應該不用每一層都做， 圖片上的normalize還有dropout在PositionwiseFeedForward"""
        if self.cfg.SCALE_EMB:
            input_vec *= self.cfg.D_MODEL_LIST[0] ** 0.5
        input_vec = self.dropout(self.position_enc(input_vec))
        enc_output  = self.layer_norm(input_vec)

        for enc_layer in self.encoder_stack:
            """
            input_vec: (batch, text_len, word_emb_len)
            src_mask: (batch, text_len) ==> unsqueeze(-2) ==> (batch, 1, text_len)
            """
            enc_output , enc_slf_attn = enc_layer(enc_output , slf_attn_mask=src_mask.unsqueeze(-2))
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
            
        return enc_output

class Discriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.position_enc = PositionalEncoding(cfg.D_WORD_VEC, n_position=cfg.N_POSITION)
        self.dropout = nn.Dropout(p=cfg.DROPOUT)
        self.layer_norm = nn.LayerNorm(cfg.D_MODEL_LIST[0], eps=1e-6)
        self.encoder_stack = nn.ModuleList([
                                            EncoderLayer(d_model=cfg.D_MODEL_LIST[i], 
                                                        d_inner=cfg.D_MODEL_LIST[i] * cfg.D_INNER_SCALE, 
                                                        n_head=cfg.N_HEAD_LIST[i], 
                                                        d_k=cfg.D_K_LIST[i], 
                                                        d_v=cfg.D_V_LIST[i], 
                                                        dropout=cfg.DROPOUT)
                                                        for i in range(cfg.N_LAYERS-1, 0-1, -1)])


if __name__ == "__main__":
    a = [1,2,3,4,5]
    for i in range(4, 0-1, -1):
        #print(i)
        print(a[i])