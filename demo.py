import json
import copy
import torch
from models.model_gan import Generator
from models.transformer import Transformer, Encoder, Decoder
from config import cfg
from utils import get_noise_tensor, get_caption_vector
import string
from pycocotools.coco import COCO
import fasttext
import fasttext.util
from geometry import so32rotation, rotation2so3
import numpy as np

device = torch.device('cpu')

if __name__ == "__main__":
    checkpoint = torch.load('./models/checkpoints/epoch_16059' + ".chkpt", map_location=torch.device('cpu')) #in docker
    #checkpoint = torch.load('/media/remote_home/chang/z_master-thesis/models/epoch_284' + ".chkpt")
    #net_g = Generator(cfg.ENC_PARAM_G, cfg.DEC_PARAM_G, cfg.FC_LIST_G).to(device)
    
    shareEncoder = Encoder(n_layers=cfg.ENC_PARAM.n_layers, 
                            d_model=cfg.ENC_PARAM.d_model, 
                            d_inner_scale=cfg.ENC_PARAM.d_inner_scale, 
                            n_head=cfg.ENC_PARAM.n_head, 
                            d_k=cfg.ENC_PARAM.d_k, 
                            d_v=cfg.ENC_PARAM.d_v, 
                            dropout=cfg.ENC_PARAM.dropout, 
                            scale_emb=cfg.ENC_PARAM.scale_emb)

    decoder_g = Decoder(n_layers=cfg.DEC_PARAM_G.n_layers, 
                            d_model=cfg.DEC_PARAM_G.d_model, 
                            d_inner_scale=cfg.DEC_PARAM_G.d_inner_scale, 
                            n_head=cfg.DEC_PARAM_G.n_head, 
                            d_k=cfg.DEC_PARAM_G.d_k, 
                            d_v=cfg.DEC_PARAM_G.d_v, 
                            dropout=cfg.DEC_PARAM_G.dropout, 
                            scale_emb=cfg.DEC_PARAM_G.scale_emb)

    net_g = Generator(shareEncoder, decoder_g, cfg.FC_LIST_G).to(device)
    net_g.load_state_dict(checkpoint['model_g'])

    coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    coco_keypoint = COCO(cfg.COCO_keypoints_TRAIN)
    text_model = fasttext.load_model(cfg.TEXT_MODEL_PATH)
    fasttext.util.reduce_model(text_model, 150)
    
    with open('../eft/eft_fit/COCO2014-All-ver01.json','r') as f: # in docker
    #with open('/media/remote_home/chang/eft/eft_fit/COCO2014-All-ver01_with_caption.json','r') as f:
        eft_data = json.load(f)
        eft_data_all = eft_data['data']    
  
    
    eft_all_fake = copy.deepcopy(eft_data_all)
    eft_all_fake = eft_all_fake[:50]
    print(len(eft_all_fake))
    net_g.eval()
    
    output = []

    for i in range(len(eft_all_fake)):  
        data = {}      
        #text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
        #text_match_mask = batch.get('vec_mask').to(device)
        img_id = coco_keypoint.loadAnns(eft_all_fake[i]['annotId'])[0]['image_id']
        # 但對於同一個圖片會有很多語意相同的captions
        caption_ids = coco_caption.getAnnIds(imgIds=img_id)
        captions_ann = coco_caption.loadAnns(ids=caption_ids)[0]
        
        print(captions_ann['caption'])
        caption_without_punctuation = ''.join([i for i in captions_ann['caption'] if i not in string.punctuation])
        if len(caption_without_punctuation.split()) < cfg.MAX_SENTENCE_LEN:
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
            output.append(eft_all_fake[i])
    
    with open('./demo/eft_50.json', 'w') as f:
        json.dump(output, f)


    #print(len(eft_all_with_caption))
    #print(eft_all_with_caption[0].keys())