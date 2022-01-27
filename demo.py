import json
import copy
import torch
from models.model_gan import Generator, LinearDecoder, Discriminator
from models.transformer import Encoder, Decoder
from config import cfg
from utils import get_noise_tensor, get_caption_vector
import string
from pycocotools.coco import COCO
import fasttext
import fasttext.util
from geometry import so32rotation, rotation2so3
import numpy as np
from DAMSM import RNN_ENCODER
import pickle
import os
from tqdm import tqdm
import torch.nn.functional as F
import math
#keywords = ['frisbee','skateboard', 'tennis', 'ski']
keywords = ['baseball', 'skateboard', 'surf','ski']
not_keywords = ["stand", "sit", "walk", "observ", "parked", "picture", "photo", "post"]
device = torch.device('cpu')

def saveImgOrNot(caption):
    save_this = False
    category = 0
    for nnk in range(len(not_keywords)):
        if not_keywords[nnk] in caption:
            return save_this, category

    for nk in range(len(keywords)):
        if keywords[nk] in caption:                    
            save_this = True
            category = nk
    return save_this, category
    
def pad_text(text, d_word_vec):
    batch_s = text.size(0)
    new_text = torch.zeros((batch_s,24,d_word_vec), dtype=torch.float32)
    for i in range(batch_s):
        if len(text[i]) < 24:
            new_text[i][0:len(text[i])] = text[i]
            for j in range(len(text[i]), 24):
                new_text[i][j] = torch.zeros(d_word_vec, dtype=torch.float32).unsqueeze(0)
        
    return new_text


if __name__ == "__main__":
    checkpoint = torch.load('./models/checkpoints/epoch_9500' + ".chkpt", map_location=torch.device('cpu')) #in docker
    #checkpoint = torch.load('/media/remote_home/chang/z_master-thesis/models/checkpoints/epoch_9' + ".chkpt")
    ##
    ## model_gan 得生成器有手寫devise判讀 要手動改 docker時因為不能用CUDA所以沒問題
    ##
    shareEncoder = Encoder(n_layers=cfg.ENC_PARAM_D.n_layers, 
                            d_model=72, 
                            d_inner_scale=cfg.ENC_PARAM_D.d_inner_scale, 
                            n_head=cfg.ENC_PARAM_D.n_head, 
                            d_k=cfg.ENC_PARAM_D.d_k, 
                            d_v=cfg.ENC_PARAM_D.d_v, 
                            dropout=cfg.ENC_PARAM_D.dropout, 
                            scale_emb=cfg.ENC_PARAM_D.scale_emb)
    decoder_g = LinearDecoder(72)    
    decoder_d = Decoder(n_layers=cfg.DEC_PARAM_D.n_layers, 
                            d_model=72, 
                            d_inner_scale=cfg.DEC_PARAM_D.d_inner_scale, 
                            n_head=cfg.DEC_PARAM_D.n_head, 
                            d_k=cfg.DEC_PARAM_D.d_k, 
                            d_v=cfg.DEC_PARAM_D.d_v, 
                            dropout=cfg.DEC_PARAM_D.dropout, 
                            scale_emb=cfg.DEC_PARAM_D.scale_emb)
                            
    net_g = Generator(decoder_g, device, cfg.D_WORD_VEC, cfg.NOISE_WEIGHT_G).to(device)
    net_d = Discriminator(shareEncoder, decoder_d, device, cfg.D_WORD_VEC, cfg.NOISE_WEIGHT_D).to(device)
    net_g.load_state_dict(checkpoint['model_g'])
    net_d.load_state_dict(checkpoint['model_d'])

    coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    coco_keypoint = COCO(cfg.COCO_keypoints_TRAIN)
    text_encoder = RNN_ENCODER()
    state_dict = torch.load('./coco/text_encoder100.pth', map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.to(device)
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()   
    print("Text model loaded")
    text_model = text_encoder
    
    with open('../eft/eft_fit/COCO2014-All-ver01.json','r') as f: # in docker
    #with open('/media/remote_home/chang/eft/eft_fit/COCO2014-All-ver01.json','r') as f:
        eft_data = json.load(f)
  
    eft_data = eft_data['data'] 
    #eft_all_with_caption = eft_all_with_caption
    print(len(eft_data))
    net_g.eval()
    
    output = []
    min_list = []
    max_list = []
    previous_ids = []
    filenamepath = os.path.join('./coco', 'filenames.pickle')    
    with open(filenamepath, 'rb') as f:
        filenames = np.array(pickle.load(f))
    captionpath = os.path.join('./coco', 'captions.pickle')
    
    with open(captionpath, 'rb') as f:
        x = pickle.load(f)
        captions = np.array(x[0])
    for i in tqdm(range(len(eft_data)), desc='  - (Dataset)   ', leave=False):
        if i % 8 != 0:
            continue
        data = {}      
        #text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
        #text_match_mask = batch.get('vec_mask').to(device)
        img_id = coco_keypoint.loadAnns(eft_data[i]['annotId'])[0]['image_id']
        keypoint_ids = coco_keypoint.getAnnIds(imgIds=img_id)
        if img_id in previous_ids:
            continue
        if len(keypoint_ids) != 1:
            continue
        else:
            previous_ids.append(img_id)
        # 但對於同一個圖片會有很多語意相同的captions
        caption_ids = coco_caption.getAnnIds(imgIds=img_id)
        captions_anns = coco_caption.loadAnns(ids=caption_ids)

        save_this, category = saveImgOrNot(captions_anns[0]['caption'])
        if not save_this:
            continue
        
        
        kk = np.where(filenames == eft_data[i]['imageName'][:-4]) # remove ".jpg"
        if True:
            new_sent_ix = kk[0][0] * 5 + 1
            caption = captions[new_sent_ix]
            caption = torch.tensor(caption).unsqueeze(0)
            caption_len = torch.tensor(len(caption[0])).unsqueeze(0)
            if 10-caption_len <= 0:
                continue
            mask = []
            for _ in range(caption_len):
                mask.append(1)
            for _ in range(caption_len, 24):
                mask.append(0)
            mask = torch.tensor(mask).unsqueeze(0)
            hidden = text_model.init_hidden(1)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, _ = text_model(caption, caption_len, hidden)
            caption_emb = words_embs.detach().to(device)
            caption_emb = pad_text(caption_emb, cfg.D_WORD_VEC).to(device)
            noise = torch.randn((1, 24, cfg.D_WORD_VEC), dtype=torch.float32).to(device)
            
            enc_output = net_d.encoder(net_d.compress(caption_emb).detach(), mask).detach()
            so3_fake = net_g(enc_output, mask, noise).detach()
            """
            norm = torch.norm(gg[:,:,:-1], p=2, dim=-1, keepdim=False).unsqueeze(-1)+0.0000000001
            scale = F.hardtanh(gg[:,:,-1:], min_val=-math.pi/2.0 , max_val=math.pi/2.0) +  math.pi/2.0 + 0.0000000001
            so3_fake = gg[:,:,:-1] * torch.div(scale, norm).repeat(1,1,3)"""
        
            so3_fake = so3_fake[0].detach().numpy()
            parm_pose = []
            for jjj in range(len(so3_fake)):
                parm_pose.append(so32rotation(so3_fake[jjj]))
                
            """for j in range(len(eft_all_fake[i]['parm_pose'])):
                print(rotation2so3(eft_all_fake[i]['parm_pose'][j]))
            print("=================================")"""
            """v = []
            for j in range(len(eft_all_fake[i]['parm_pose'])):
                v.append(so32rotation(rotation2so3(eft_all_fake[i]['parm_pose'][j])))"""
            eft_data[i]['parm_pose'] = parm_pose
            eft_data[i]['caption'] = captions_anns[0]['caption']
            if 10-caption_len >= 0:
                print(caption_len)
                output.append(eft_data[i])
    
    with open('./demo/eft_50.json', 'w') as f:
        json.dump(output, f)
    
