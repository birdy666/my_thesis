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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    checkpoint = torch.load('./models/epoch_284' + ".chkpt") #in docker
    #checkpoint = torch.load('/media/remote_home/chang/z_master-thesis/models/epoch_284' + ".chkpt")
    net_g = Generator(cfg.ENC_PARAM_G, cfg.DEC_PARAM_G, cfg.FC_LIST_G).to(device)
    net_g.load_state_dict(checkpoint['model_g'])

    coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    text_model = fasttext.load_model(cfg.TEXT_MODEL_PATH)
    fasttext.util.reduce_model(text_model, 150)
    
    with open('../eft/eft_fit/COCO2014-All-ver01_with_caption.json','r') as f: # in docker
    #with open('/media/remote_home/chang/eft/eft_fit/COCO2014-All-ver01_with_caption.json','r') as f:
        eft_all_with_caption = json.load(f)
  
    
    eft_all_fake = copy.deepcopy(eft_all_with_caption)
    eft_all_fake = eft_all_fake[:20]
    print(len(eft_all_fake))
    net_g.eval()
    
    for i in range(len(eft_all_fake)):  
        data = {}      
        #text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
        #text_match_mask = batch.get('vec_mask').to(device)
        caption =  coco_caption.loadAnns(eft_all_fake[i]['annotId'])[0]['caption']
        print(caption)
        caption_without_punctuation = ''.join([i for i in caption if i not in string.punctuation])
        data['vector'], data['vec_mask'] = get_caption_vector(text_model, caption_without_punctuation, cfg.MAX_SENTENCE_LEN, cfg.D_WORD_VEC)
    
        text_match = torch.tensor(data['vector'], dtype=torch.float32).unsqueeze(0)
        text_match_mask = torch.tensor(data['vec_mask'], dtype=torch.int).unsqueeze(0)
        noise = torch.randn((1, 24, 150), dtype=torch.float32).to(device)
        
        so3_fake = net_g(text_match, text_match_mask, noise)
        
        so3_fake = so3_fake[0].detach().numpy()
        text_match = text_match.sum().unsqueeze(0)
        
        parm_pose = []
        for j in range(len(so3_fake)):
            parm_pose.append(so32rotation(so3_fake[j]))
        eft_all_fake[i]['parm_pose'] = parm_pose
    
    with open('./demo/eft_50.json', 'w') as f:
        json.dump(eft_all_fake, f)


    #print(len(eft_all_with_caption))
    #print(eft_all_with_caption[0].keys())