import json
import copy
import torch
from models.model_gan import Generator
from config import cfg
from utils import get_noise_tensor, get_caption_vector
import string
from pycocotools.coco import COCO
import fasttext
import fasttext.util
from geometry import so32rotation, rotation2so3
import numpy as np

import torch.nn.functional as F
import math

device = torch.device('cpu')

if __name__ == "__main__":
    checkpoint = torch.load('./models/checkpoints/epoch_24' + ".chkpt", map_location=torch.device('cpu')) #in docker
    #checkpoint = torch.load('/media/remote_home/chang/z_master-thesis/models/checkpoints/epoch_9' + ".chkpt")
    ##
    ## model_gan 得生成器有手寫devise判讀 要手動改 docker時因為不能用CUDA所以沒問題
    ##
    net_g = Generator(cfg.ENC_PARAM_G, cfg.DEC_PARAM_G, cfg.FC_LIST_G, cfg.D_WORD_VEC).to(device)
    net_g.load_state_dict(checkpoint['model_g'])

    coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    coco_keypoint = COCO(cfg.COCO_keypoints_TRAIN)
    text_model = fasttext.load_model(cfg.TEXT_MODEL_PATH)
    fasttext.util.reduce_model(text_model, cfg.D_WORD_VEC)
    
    with open('../eft/eft_fit/COCO2014-All-ver01.json','r') as f: # in docker
    #with open('/media/remote_home/chang/eft/eft_fit/COCO2014-All-ver01.json','r') as f:
        eft_all_with_caption = json.load(f)
  
    eft_all_fake = eft_all_with_caption['data'] 
    eft_all_fake = eft_all_fake[:10]
    print(len(eft_all_fake))
    net_g.eval()
    
    output = []
    min_list = []
    max_list = []
    previous_ids = []
    for i in range(len(eft_all_fake)):
        """if i % 10 !=0:
            continue"""
        data = {}      
        #text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
        #text_match_mask = batch.get('vec_mask').to(device)
        img_id = coco_keypoint.loadAnns(eft_all_fake[i]['annotId'])[0]['image_id']
        
        # 但對於同一個圖片會有很多語意相同的captions
        caption_ids = coco_caption.getAnnIds(imgIds=img_id)
        captions_anns = coco_caption.loadAnns(ids=caption_ids)

        for j, captions_ann in enumerate(captions_anns):
            if j > 0:
                break        
            print(captions_ann['caption'])
            caption_without_punctuation = ''.join([j for j in captions_ann['caption'] if j not in string.punctuation])
            if len(caption_without_punctuation.split()) < cfg.MAX_SENTENCE_LEN:
                data['vector'], data['vec_mask'] = get_caption_vector(text_model, caption_without_punctuation, cfg.MAX_SENTENCE_LEN, cfg.D_WORD_VEC)

                text_match = torch.tensor(data['vector'], dtype=torch.float32).unsqueeze(0)
                text_match_mask = torch.tensor(data['vec_mask'], dtype=torch.int).unsqueeze(0)
                noise = torch.randn((1, 24, cfg.D_WORD_VEC), dtype=torch.float32).to(device)

                gg = net_g(text_match, text_match_mask, noise)
                norm = torch.norm(gg[:,:,:-1], p=2, dim=-1, keepdim=False).unsqueeze(-1)+0.0000000001
                scale = F.hardtanh(gg[:,:,-1:], min_val=-math.pi/2.0 , max_val=math.pi/2.0) +  math.pi/2.0 + 0.0000000001
                so3_fake = gg[:,:,:-1] * torch.div(scale, norm).repeat(1,1,3)
        
                so3_fake = so3_fake[0].detach().numpy()
                parm_pose = []
                for j in range(len(so3_fake)):
                    parm_pose.append(so32rotation(so3_fake[j]))
                
                for j in range(len(eft_all_fake[i]['parm_pose'])):
                    print(rotation2so3(eft_all_fake[i]['parm_pose'][j]))
                print("=================================")
                """v = []
                for j in range(len(eft_all_fake[i]['parm_pose'])):
                    v.append(so32rotation(rotation2so3(eft_all_fake[i]['parm_pose'][j])))"""
                eft_all_fake[i]['parm_pose'] = parm_pose
                output.append(eft_all_fake[i])
    
    with open('./demo/eft_50.json', 'w') as f:
        json.dump(output, f)
    
