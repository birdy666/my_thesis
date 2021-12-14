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
import math


from geometry import rotation2so3
from utils import get_caption_vector
from config import cfg

#keywords = ['ski', 'baseball', 'motor','tennis','skateboard','kite']
#keywords = ['frisbee', 'baseball', 'skateboard', 'surf','skiing']
keywords = ['frisbee','skateboard', 'tennis']
not_keywords = ["stand", "sit", "walk", "observ", "parked", "picture", "photo", "post"]

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
    if  cfg.D_WORD_VEC < 300:
        fasttext.util.reduce_model(text_model, cfg.D_WORD_VEC)
    print("Text model loaded")
    # load eft data
    eft_data_all = getEFTCaption(cfg)        
    # get the dataset (single person, with captions)
    train_size = int(len(eft_data_all))
    print("dataset size: ", train_size)
    print("Creating dataset_train")
    dataset_train = TheDataset(cfg, eft_data_all[:int(train_size*0.9)], coco_caption, coco_keypoint, text_model=text_model)
    print("Creating dataset_val")
    dataset_val = TheDataset(cfg, eft_data_all[int(train_size*0.9):train_size], coco_caption, coco_keypoint, text_model=text_model)
    print("Datasets created")
    #return text_model, dataset, dataset_val, data_loader#, text_match_val, label_val
    return text_model, eft_data_all, dataset_train, dataset_val

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
    

class TheDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, eft_data_all, coco_caption, coco_keypoint, text_model=None, val=False):
        self.dataset = []
        self.cfg = cfg
        previous_img_ids = []
        print(math.pi)
        for i in tqdm(range(len(eft_data_all)), desc='  - (Dataset)   ', leave=False):            
            # one eft data correspond to one keypoint in one img
            img_id = coco_keypoint.loadAnns(eft_data_all[i]['annotId'])[0]['image_id']
            if img_id in previous_img_ids:
                continue
            else:
                previous_img_ids.append(img_id)
            # many captions for one img
            caption_ids = coco_caption.getAnnIds(imgIds=img_id)
            captions_anns = coco_caption.loadAnns(ids=caption_ids)
            
            save_this, category = saveImgOrNot(captions_anns[0]['caption'])
            if not save_this:
                continue
            # n caption with the same pose
            for j, caption_ann in enumerate(captions_anns):
                if j > 1:
                    break
                data = {'caption': caption_ann['caption'],
                        'parm_pose': eft_data_all[i]['parm_pose'],
                        'parm_shape': eft_data_all[i]['parm_shape'],
                        'smpltype': eft_data_all[i]['smpltype'],
                        'annotId': eft_data_all[i]['annotId'],
                        'imageName': eft_data_all[i]['imageName'],
                        'category': category}
                data['so3'] = np.array([rotation2so3(R) for R in data['parm_pose']])
                # add sentence encoding
                if text_model is not None:
                    caption_without_punctuation = ''.join([i for i in data['caption'] if i not in string.punctuation])
                    if len(caption_without_punctuation.split()) < cfg.MAX_SENTENCE_LEN:
                        data['vector'], data['vec_mask'] = get_caption_vector(text_model, caption_without_punctuation, cfg.JOINT_NUM, cfg.D_WORD_VEC)
                        self.dataset.append(data)      
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        item = dict()
        item['so3'] = torch.tensor(data['so3'], dtype=torch.float32)
        item['vector'] = torch.tensor(data['vector'], dtype=torch.float32)
        item['vec_mask'] = torch.tensor(data['vec_mask'], dtype=torch.float32)
        item['vec_interpolated'], item['vec_interpolated_mask'] = self.get_interpolated_text(0.5)
        item['so3_wrong']= torch.tensor(self.get_so3_wrong(index), dtype=torch.float32)
        return item
    
    # get a batch of random so3 from the whole dataset    
    def get_so3_wrong(self, index):
        """others = list(range(0, index-5)) + list(range(index+1+5, len(self.dataset)))
        data_random = self.dataset[random.choice(others)]
        return data_random['so3']"""
        while True:
            data_random = self.dataset[random.choice(list(range(0, len(self.dataset))))]
            if data_random['category'] != self.dataset[index]['category']:
                return data_random['so3']
        """data_randoms = [self.dataset[random.choice(others)]['so3'] for i in range(40)]
        return np.sum(data_randoms, axis=0)/40"""

    # get a batch of random interpolated caption sentence vectors from the whole dataset
    # not sure if it makes sense to use "or" for mask
    def get_interpolated_text(self, beta, f_mask = lambda x,y: [1 if x[i] or y[i] else 0 for i in range(len(x))]):
        index1 = random.randint(0, len(self.dataset)-1)  # randint(a, b) （a <= N <= b）
        index2 = random.randint(0, len(self.dataset)-1)
        vector1 = self.dataset[index1]['vector']
        vector2 = self.dataset[index2]['vector']
        mask = f_mask(self.dataset[index1]['vec_mask'], self.dataset[index2]['vec_mask'])
        # interpolate caption sentence vectors
        return beta * vector1 + (1 - beta) * vector2, np.array(mask)