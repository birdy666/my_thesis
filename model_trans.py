import torch
import torch.nn as nn

from transformer import PositionalEncoding, EncoderLayer

def getModel(cfg):
    return Generator(cfg)

class Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.position_enc = PositionalEncoding(cfg.D_WORD_VEC, n_position=cfg.N_POSITION)
        self.dropout = nn.Dropout(p=cfg.DROPOUT)
        
        self.encoder_stack = nn.ModuleList([
                                            EncoderLayer(d_model=cfg.D_MODEL_LIST[i], 
                                                        d_inner=cfg.D_MODEL_LIST[i] * cfg.D_INNER_SCALE, 
                                                        n_head=cfg.N_HEAD_LIST[i], 
                                                        d_k=cfg.D_K_LIST[i], 
                                                        d_v=cfg.D_V_LIST[i], 
                                                        dropout=cfg.DROPOUT)
                                                        for i in range(cfg.N_LAYERS)])
        
        self.layer_norm = nn.LayerNorm(cfg.D_MODEL_LIST[0], eps=1e-6)
    
    """不知道mask要傳甚麼嗚嗚 TODO"""
    def forward(self, input_vec, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        input_vec = input_vec.copy()
        """這裡應該不用每一層都做， 圖片上的normalize還有dropout在PositionwiseFeedForward"""
        if self.cfg.SCALE_EMB:
            input_vec *= self.cfg.D_MODEL_LIST[0] ** 0.5
        input_vec = self.dropout(self.position_enc(input_vec))
        input_vec = self.layer_norm(input_vec)

        for enc_layer in self.encoder_stack:
            input_vec, enc_slf_attn = enc_layer(input_vec, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        """純粹覺得return叫input很怪"""
        enc_output = input_vec
        return enc_output

class Discriminator(nn.Module):
    def __init__(self, cfg):
        """連架構都還沒想好R 感覺這裡是不是相較隨興 反正中間加入word vec最後輸出Y/N 就好惹"""
        pass


if __name__ == "__main__":
    a = [1,2,3,4]
    print([i*2 for i in a])