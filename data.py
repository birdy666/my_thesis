import fasttext
import fasttext.util
import numpy as np
from pycocotools.coco import COCO
import torch
import json
import os
import random
from tqdm import tqdm
#import cv2
import string


from geometry import rotation2so3
from utils import get_noise_tensor, get_caption_vector
from config import cfg


#data['parm_pose']     #24x3x3, 3D rotation matrix for 24 joints
    #data['parm_shape']       #10 dim vector
    #data['parm_cam']        #weak-perspective projection: [cam_scale, cam_transX, cam_tranY]
    #data['bbox_scale']     #float
    #data['bbox_center']    #[x,y]
    #data['joint_validity_openpose18']      #Joint validity in openpose18 joint order
    #data['smpltype']           #smpl or smplx
    #data['annotId']            #Only for COCO dataset. COCO annotation ID
    #data['imageName']          #image name (basename only)

    #data['subjectId']          #(optional) a unique id per sequence. usually {seqName}_{id}
    #Load EFT fitting data

def getEFTCaption(cfg):
    with open(cfg.EFT_FIT_PATH,'r') as f:
        eft_data = json.load(f)
        print("EFT data: ver {}".format(eft_data['ver']))
        eft_data_all = eft_data['data']    
    return eft_data_all

def getData(cfg):
    # load coco  
    coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    coco_keypoint = COCO(cfg.COCO_keypoints_TRAIN)
    # load text model
    print("Loading text model")
    text_model = fasttext.load_model(cfg.TEXT_MODEL_PATH)  
    fasttext.util.reduce_model(text_model, cfg.D_WORD_VEC)
    print("Text model loaded")
    # load eft data
    eft_data_all = getEFTCaption(cfg)        
    # get the dataset (single person, with captions)
    train_size = int(len(eft_data_all)*0.2*0.1)
    print("dataset size: ", train_size)
    print("Creating dataset_train")
    dataset_train = TheDataset(cfg, eft_data_all[:int(train_size*0.9)], coco_caption, coco_keypoint, text_model=text_model)
    print("Creating dataset_val")
    dataset_val = TheDataset(cfg, eft_data_all[int(train_size*0.9):train_size], coco_caption, coco_keypoint, text_model=text_model)
    print("Datasets created")
    #return text_model, dataset, dataset_val, data_loader#, text_match_val, label_val
    return text_model, eft_data_all, dataset_train, dataset_val


class TheDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, eft_data_all, coco_caption, coco_keypoint, text_model=None, val=False):
        self.dataset = []
        self.cfg = cfg
        previous_ids = []
        for i in tqdm(range(len(eft_data_all)), desc='  - (Dataset)   ', leave=False):            
            # 一筆eft資料對應到一張img中的一筆keypoint
            img_id = coco_keypoint.loadAnns(eft_data_all[i]['annotId'])[0]['image_id']
            
            # 但對於同一個圖片會有很多語意相同的captions
            caption_ids = coco_caption.getAnnIds(imgIds=img_id)
            captions_anns = coco_caption.loadAnns(ids=caption_ids)
            # 每個cation都創一個資料
            for j, caption_ann in enumerate(captions_anns):
                if j > 1:
                    break
                data = {'caption': caption_ann['caption'],
                        'parm_pose': eft_data_all[i]['parm_pose'],
                        'parm_shape': eft_data_all[i]['parm_shape'],
                        'smpltype': eft_data_all[i]['smpltype'],
                        'annotId': eft_data_all[i]['annotId'],
                        'imageName': eft_data_all[i]['imageName']}
                data['so3'] = np.array([rotation2so3(R) for R in data['parm_pose']])
                # add sentence encoding
                if text_model is not None:
                    caption_without_punctuation = ''.join([i for i in data['caption'] if i not in string.punctuation])
                    if len(caption_without_punctuation.split()) < cfg.MAX_SENTENCE_LEN:
                        data['vector'], data['vec_mask'] = get_caption_vector(text_model, caption_without_punctuation, cfg.MAX_SENTENCE_LEN, cfg.D_WORD_VEC)
                        self.dataset.append(data)      
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        item = dict()        
        # change heatmap range from [0,1] to[-1,1]
        item['so3'] = torch.tensor(data['so3'], dtype=torch.float32)
        #item['vector'].unsqueeze_(-1).unsqueeze_(-1) 完全不知道我這裡要不要
        item['vector'] = torch.tensor(data['vector'], dtype=torch.float32)
        item['vec_mask'] = torch.tensor(data['vec_mask'], dtype=torch.int)
        """unsqueeze_ is in place operation, unsqueeze isn't"""
        item['vec_mismatch'], item['vec_mismatch_mask'] = self.get_text_mismatch(index)
        item['vec_interpolated'], item['vec_interpolated_mask'] = self.get_interpolated_text(0.5)
            
        return item

    # get a batch of random caption sentence vectors from the whole dataset
    def get_text_mismatch(self, index):
        others = list(range(0, index)) + list(range(index+1, len(self.dataset)))
        data_random = self.dataset[random.choice(others)]
        return data_random['vector'], data_random['vec_mask']
    
    # get a batch of random interpolated caption sentence vectors from the whole dataset
    # 預設mask我是用or但不確定合不合理?
    def get_interpolated_text(self, beta, f_mask = lambda x,y: [1 if x[i] or y[i] else 0 for i in range(len(x))]):
        index1 = random.randint(0, len(self.dataset)-1)  # randint(a, b) （a <= N <= b）
        index2 = random.randint(0, len(self.dataset)-1)
        vector1 = self.dataset[index1]['vector']
        vector2 = self.dataset[index2]['vector']
        mask = f_mask(self.dataset[index1]['vec_mask'], self.dataset[index2]['vec_mask'])
        # interpolate caption sentence vectors
        return beta * vector1 + (1 - beta) * vector2, np.array(mask)


if __name__ == "__main__":
    """coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    coco_keypoint = COCO(cfg.COCO_keypoints_TRAIN)
    generate_eft(cfg, coco_caption, coco_keypoint)"""
    """#with open('../eft/eft_fit/COCO2014-All-ver01_with_caption.json','r') as f: # in docker
    with open('/media/remote_home/chang/eft/eft_fit/COCO2014-All-ver01_with_caption.json','r') as f:
        eft_all_with_caption = json.load(f)
    coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    coco_keypoint = COCO(cfg.COCO_keypoints_TRAIN)
    for i in range(1):
        print(eft_all_with_caption[i]['imageName'])
        imgid = coco_keypoint.loadAnns(eft_all_with_caption[17]['annotId'])[0]['image_id']
        caption_ids = coco_caption.getAnnIds(imgIds=imgid)
        captions = coco_caption.loadAnns(ids=caption_ids)
        print(captions)"""
    #coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    #coco_keypoint = COCO(cfg.COCO_keypoints_TRAIN)
    eft_data_all = getEFTCaption(cfg)  
    min_list = []
    max_list = []      
    for i in range(1000):
        """print("=======================================================")
        print(str(i))
        print("=======================================================")"""
        so3 = np.array([rotation2so3(R) for R in eft_data_all[i]['parm_pose']])
        norm = np.linalg.norm(so3, axis=1)
        min_list.append(min(norm))
        max_list.append(max(norm))
        """# 一筆eft資料只有一筆img_id
        img_id = coco_keypoint.loadAnns(eft_data_all[i]['annotId'])[0]['image_id']
        # 但對於同一個圖片會有很多語意相同的captions
        caption_ids = coco_caption.getAnnIds(imgIds=img_id)
        captions_anns = coco_caption.loadAnns(ids=caption_ids)
        # 每個cation都創一個資料
        for j, caption_ann in enumerate(captions_anns):
            if j > 3:
                    break
            print(caption_ann['caption'])"""
        
    print(min(min_list))
    print(max(max_list))