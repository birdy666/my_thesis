import json
import copy
import torch
from models.model_gan import Generator
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
keywords = ['baseball', 'skateboard', 'surf','ski', 'motor','tennis','frisbee', "sit",'kite']
not_keywords = ["stand", "sit", "walk", "observ", "parked", "picture", "photo", "post"]
device = torch.device('cpu')
import copy

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
    checkpoint = torch.load('./models/checkpoints/epoch_120' + ".chkpt", map_location=torch.device('cpu')) #in docker
    #checkpoint = torch.load('/media/remote_home/chang/z_master-thesis/models/checkpoints/epoch_9' + ".chkpt")
    ##
    ## model_gan 得生成器有手寫devise判讀 要手動改 docker時因為不能用CUDA所以沒問題
    ##
    encoder_g = Encoder(n_layers=cfg.ENC_PARAM_G.n_layers, 
                            d_model=cfg.ENC_PARAM_G.d_model, 
                            d_inner_scale=cfg.ENC_PARAM_G.d_inner_scale, 
                            n_head=cfg.ENC_PARAM_G.n_head, 
                            d_k=cfg.ENC_PARAM_G.d_k, 
                            d_v=cfg.ENC_PARAM_G.d_v, 
                            dropout=cfg.ENC_PARAM_G.dropout, 
                            scale_emb=cfg.ENC_PARAM_G.scale_emb)
    decoder_g = Decoder(n_layers=cfg.DEC_PARAM_G.n_layers, 
                            d_model=cfg.DEC_PARAM_G.d_model, 
                            d_inner_scale=cfg.DEC_PARAM_G.d_inner_scale, 
                            n_head=cfg.DEC_PARAM_G.n_head, 
                            d_k=cfg.DEC_PARAM_G.d_k, 
                            d_v=cfg.DEC_PARAM_G.d_v, 
                            dropout=cfg.DEC_PARAM_G.dropout, 
                            scale_emb=cfg.DEC_PARAM_G.scale_emb)    
    
                            
    net_g = Generator(encoder_g,decoder_g, device, cfg.D_WORD_VEC, cfg.NOISE_WEIGHT_G).to(device)
    
    net_g.load_state_dict(checkpoint['model_g'])

    coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    coco_keypoint = COCO(cfg.COCO_keypoints_TRAIN)
    
    
    with open('../eft/eft_fit/COCO2014-All-ver01.json','r') as f: # in docker
    #with open('/media/remote_home/chang/eft/eft_fit/COCO2014-All-ver01.json','r') as f:
        eft_data = json.load(f)
  
    eft_data = eft_data['data'] 
    #eft_all_with_caption = eft_all_with_caption
    net_g.eval()
    output_real = []
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
        if i > 20000:
            break
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
            caption_len = len(caption)
            
            if 10-caption_len <= 0:
                continue
            mask = []
            for _ in range(caption_len):
                mask.append(1)
            for _ in range(caption_len, 24):
                mask.append(0)
            mask = torch.tensor(mask).unsqueeze(0)
            caption = np.pad(caption, (0, 24-len(caption)))
            caption = torch.tensor(caption).unsqueeze(0)
            noise = get_noise_tensor(1, cfg.NOISE_SIZE).to(device)
            
            so3_fake = net_g(caption, mask, noise).detach()
            
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
            
            if 10-caption_len >= 0:
                output_real.append(copy.deepcopy(eft_data[i]))
                eft_data[i]['parm_pose'] = parm_pose
                eft_data[i]['caption'] = captions_anns[0]['caption']
                output.append(eft_data[i])
    
    with open('./demo/eft_50.json', 'w') as f:
        json.dump(output, f)
    with open('./demo/eft_50_real.json', 'w') as f:
        json.dump(output_real, f)
    
