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
import pickle

from DAMSM import RNN_ENCODER
from geometry import rotation2so3
from utils import get_caption_vector, get_caption_mask
from config import cfg

#keywords = ['ski', 'baseball', 'motor','tennis','skateboard','kite']
#keywords = ['frisbee', 'baseball', 'skateboard', 'surf','skiing']
keywords = ['baseball', 'skateboard', 'surf','ski']
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

def get_textModel(cfg, device):
    # load text model
    print("Loading text model")
    """
    text_model = fasttext.load_model(cfg.TEXT_MODEL_PATH) 
    if  cfg.D_SENTENCE_VEC < 300:
        fasttext.util.reduce_model(text_model, cfg.D_SENTENCE_VEC)
    """
    text_encoder = RNN_ENCODER()
    state_dict = torch.load('/media/remote_home/chang/z_master-thesis/coco/text_encoder100.pth', map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.to(device)

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()   
    print("Text model loaded")
    return text_encoder

def getData(cfg, device):
    # load coco  
    coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    coco_keypoint = COCO(cfg.COCO_keypoints_TRAIN)
    # load text model
    text_model = get_textModel(cfg, device)
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
        ###########################################
        filenamepath = os.path.join('/media/remote_home/chang/z_master-thesis/coco', 'filenames.pickle')    
        with open(filenamepath, 'rb') as f:
            self.filenames = np.array(pickle.load(f))
        ###########################################
        captionpath = os.path.join('/media/remote_home/chang/z_master-thesis/coco', 'captions.pickle')
    
        with open(captionpath, 'rb') as f:
            x = pickle.load(f)
            self.captions = np.array(x[0])
        ##########################################    
        print(math.pi)
        for i in tqdm(range(len(eft_data_all)), desc='  - (Dataset)   ', leave=False):            
            # one eft data correspond to one keypoint in one img
            keypoint_ann = coco_keypoint.loadAnns(eft_data_all[i]['annotId'])[0]
            img_id = keypoint_ann['image_id']
            keypoint_ids = coco_keypoint.getAnnIds(imgIds=img_id)
            if img_id in previous_img_ids:
                continue
            if len(keypoint_ids) != 1:
                continue
            else:
                previous_img_ids.append(img_id)
            # many captions for one img
            caption_ids = coco_caption.getAnnIds(imgIds=img_id)
            captions_anns = coco_caption.loadAnns(ids=caption_ids)
            
            save_this, category = saveImgOrNot(captions_anns[0]['caption'])
            if not save_this:
                continue            
            
            kk = np.where(self.filenames == eft_data_all[i]['imageName'][:-4]) # remove ".jpg"
            for gg in range(2):  
                data = {'parm_pose': eft_data_all[i]['parm_pose'],
                        'parm_shape': eft_data_all[i]['parm_shape'],
                        'smpltype': eft_data_all[i]['smpltype'],
                        'annotId': eft_data_all[i]['annotId'],
                        'imageName': eft_data_all[i]['imageName'],
                        'category': category}
                data['rot_vec'] = np.array([rotation2so3(R) for R in data['parm_pose']])              
                # 82783個filename 每個file有5個caption
                new_sent_ix = kk[0][0] * 5 + gg
                data['caption'] =  self.captions[new_sent_ix]
                caption_len = len(data['caption'])  
                if 10-caption_len >= 0:       
                    mask = []
                    for _ in range(len(data['caption'])):
                        mask.append(1)
                    for _ in range(len(data['caption']), 24):
                        mask.append(0)
                    mask = np.array(mask)
                    data['caption'] = np.pad(data['caption'], (0, 24-caption_len))
                    data['caption'] = np.expand_dims(data['caption'], axis=0).transpose(1,0)
                    data['caption_len'] = caption_len
                    data['caption_mask'] = mask
                    self.dataset.append(data)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        item = dict()
        item['rot_vec'] = torch.tensor(data['rot_vec'], dtype=torch.float32)
        item['rot_vec_wrong']= torch.tensor(self.get_rot_vec_wrong(index), dtype=torch.float32)
        item['caption'] = torch.tensor(data['caption'])
        item['caption_len'] = torch.tensor(data['caption_len'])
        item['caption_mask'] = torch.tensor(data['caption_mask'])
        return item
    
    def get_caption_index(self, imageName):
        kk = np.where(self.filenames == imageName[:-4]) # remove ".jpg"
        sent_ix = random.randint(0, 5-1)
        # 82783個filename 每個file有5個caption
        new_sent_ix = kk[0][0] * 5 + sent_ix
        caption_index = self.captions[new_sent_ix]
        if 10-len(caption_index) < 0:
            print(gfhfghfg)
        caption_len = len(caption_index)        
        mask = []
        for _ in range(len(caption_index)):
            mask.append(1)
        for _ in range(len(caption_index), 24):
            mask.append(0)
        mask = np.array(mask)
        caption_index = np.pad(caption_index, (0, 24-len(caption_index)))
        caption_index = np.expand_dims(caption_index, axis=0)

        return torch.tensor(caption_index.transpose(1,0)), torch.tensor(caption_len), torch.tensor(mask)
    
    # get a batch of random rot_vec from the whole dataset    
    def get_rot_vec_wrong(self, index):
        """others = list(range(0, index-5)) + list(range(index+1+5, len(self.dataset)))
        data_random = self.dataset[random.choice(others)]
        return data_random['rot_vec']"""
        while True:
            data_random = self.dataset[random.choice(list(range(0, len(self.dataset))))]
            if data_random['category'] != self.dataset[index]['category']:
                return data_random['rot_vec']
        """data_randoms = [self.dataset[random.choice(others)]['rot_vec'] for i in range(40)]
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