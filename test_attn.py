import json
import copy
import torch
from models.model_gan import Generator, Discriminator
from models.transformer import Decoder
from config import cfg
from utils import get_noise_tensor
from pycocotools.coco import COCO
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


def output_attn(attn_g, attn_d, text, i):
    with open("./attns/attention_test"+ str(i) +".txt", "w") as fhandle:
        fhandle.write(f'{text}\n')
    with open('./attns/attention_test'+ str(i) +'.json', 'w') as outfile:
        attn_g = np.around(attn_g[0].detach().cpu().numpy().astype(float),3).tolist()
        attn_d = np.around(attn_d[0].detach().cpu().numpy().astype(float),3).tolist()
        json.dump({'attn_g':attn_g, 'attn_d':attn_d}, outfile)

if __name__ == "__main__":
    checkpoint = torch.load('./models/checkpoints/epoch_306' + ".chkpt", map_location=torch.device('cpu')) #in docker
    #checkpoint = torch.load('/media/remote_home/chang/z_master-thesis/models/checkpoints/epoch_9' + ".chkpt")
    ##
    ## model_gan 得生成器有手寫devise判讀 要手動改 docker時因為不能用CUDA所以沒問題
    
    decoder_g = Decoder(n_layers=cfg.DEC_PARAM_G.n_layers, 
                            d_model=cfg.DEC_PARAM_G.d_model, 
                            d_inner_scale=cfg.DEC_PARAM_G.d_inner_scale, 
                            n_head=cfg.DEC_PARAM_G.n_head, 
                            d_k=cfg.DEC_PARAM_G.d_k, 
                            d_v=cfg.DEC_PARAM_G.d_v, 
                            dropout=cfg.DEC_PARAM_G.dropout, 
                            scale_emb=cfg.DEC_PARAM_G.scale_emb,G=True)   
    decoder_d = Decoder(n_layers=cfg.DEC_PARAM_D.n_layers, 
                            d_model=cfg.DEC_PARAM_D.d_model, 
                            d_inner_scale=cfg.DEC_PARAM_D.d_inner_scale, 
                            n_head=cfg.DEC_PARAM_D.n_head, 
                            d_k=cfg.DEC_PARAM_D.d_k, 
                            d_v=cfg.DEC_PARAM_D.d_v, 
                            dropout=cfg.DEC_PARAM_D.dropout, 
                            scale_emb=cfg.DEC_PARAM_D.scale_emb) 
    
                            
    net_g = Generator(decoder_g, cfg.D_WORD_VEC).to(device)
    net_d = Discriminator(decoder_d, cfg.D_WORD_VEC, cfg.NOISE_WEIGHT_D, device).to(device)
    
    net_g.load_state_dict(checkpoint['model_g'])
    net_d.load_state_dict(checkpoint['model_d'])

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
    for i in tqdm(range(400,800), desc='  - (Dataset)   ', leave=True):
        
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

        """save_this, category = saveImgOrNot(captions_anns[0]['caption'])
        if not save_this:
            continue"""
        
        
        kk = np.where(filenames == eft_data[i]['imageName'][:-4]) # remove ".jpg"
        if True:
            new_sent_ix = kk[0][0] * 5 + 1
            caption = captions[new_sent_ix]
            caption_len = len(caption)
            text = captions_anns[1]['caption']
            if 15-caption_len <= 0:
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
            
            so3_fake, attn_g = net_g(caption, mask, noise)
            
            so3_fake = so3_fake[0].detach().numpy()
            parm_pose = []
            for jjj in range(len(so3_fake)):
                parm_pose.append(so32rotation(so3_fake[jjj]))
                
            if 10-caption_len >= 0:
                output_real.append(copy.deepcopy(eft_data[i]))
                #######################
                rot_vec_real = np.array([rotation2so3(R) for R in eft_data[i]['parm_pose']]) 
                rot_vec_real = torch.tensor(rot_vec_real, dtype=torch.float32)
                score_right, attn_d = net_d(net_d.embedding(caption), mask, rot_vec_real)
                output_attn(attn_g, attn_d, text, i)
                ######################
                eft_data[i]['parm_pose'] = parm_pose
                eft_data[i]['caption'] = captions_anns[0]['caption']
                output.append(eft_data[i])


    