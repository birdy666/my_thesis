from unicodedata import category
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
keywords = ['baseball', 'skateboard', 'surf','ski', 'motor','tennis','frisbee', "sit",'kite']
not_keywords = ["stand", "walk", "observ", "parked", "picture", "photo", "post"]

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

def getData(cfg, device):
    # load coco  
    coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    coco_keypoint = COCO(cfg.COCO_keypoints_TRAIN)    
    # load eft data
    eft_data_all = getEFTCaption(cfg)        
    # get the dataset (single person, with captions)
    train_size = int(len(eft_data_all))
    print("dataset size: ", train_size)
    print("Creating dataset_train")
    dataset_train = TheDataset(cfg, eft_data_all[:int(train_size*0.9)], coco_caption, coco_keypoint)
    print("Creating dataset_val")
    dataset_val = TheDataset(cfg, eft_data_all[int(train_size*0.9):train_size], coco_caption, coco_keypoint)
    print("Datasets created")
    #return text_model, dataset, dataset_val, data_loader#, text_match_val, label_val
    return eft_data_all, dataset_train, dataset_val

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
    def __init__(self, cfg, eft_data_all, coco_caption, coco_keypoint, val=False):
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
            
            """save_this, category = saveImgOrNot(captions_anns[0]['caption'])
            if not save_this:
                continue """
            category = 0      
            data = {'captions': [],
                    'caption_masks': [],
                    'caption_lens': [],
                    'parm_pose': eft_data_all[i]['parm_pose'],
                    'parm_shape': eft_data_all[i]['parm_shape'],
                    'smpltype': eft_data_all[i]['smpltype'],
                    'annotId': eft_data_all[i]['annotId'],
                    'imageName': eft_data_all[i]['imageName'],
                    'category': category}
            data['rot_vec'] = np.array([rotation2so3(R) for R in data['parm_pose']]) 
            kk = np.where(self.filenames == eft_data_all[i]['imageName'][:-4]) # remove ".jpg"
            for gg in range(5):                               
                # 82783個filename 每個file有5個caption
                new_sent_ix = kk[0][0] * 5 + gg
                # 這裡的self.captions是已經建好的字典 caption是對應到裡面的 index 而不是真正的字
                caption = self.captions[new_sent_ix]
                if 10-len(caption) > 0:  
                    caption = np.pad(caption, (0, 24-len(caption)))
                    #caption = np.expand_dims(caption, axis=0).transpose(1,0)  
                    data['captions'].append(caption) 
                    mask = []
                    for _ in range(len(caption)):
                        mask.append(1)
                    for _ in range(len(caption), 24):
                        mask.append(0)
                    mask = np.array(mask)
                    data['caption_masks'].append(mask)                    
                    data['caption_lens'].append(len(caption))
                    self.dataset.append(data)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        item = dict()
        item['rot_vec'] = torch.tensor(data['rot_vec'], dtype=torch.float32)
        item['rot_vec_wrong']= torch.tensor(self.get_rot_vec_wrong(index), dtype=torch.float32)
        item['caption'], item['caption_mask'] = self.get_caption(data)
        return item
    
    def get_caption(self, data):
        sent_ix = random.randint(0, len(data['captions'])-1)        
        return torch.tensor(data['captions'][sent_ix]), torch.tensor(data['caption_masks'][sent_ix])

    # get a batch of random rot_vec from the whole dataset    
    def get_rot_vec_wrong(self, index):
        while True:
            data_random = self.dataset[random.choice(list(range(0, len(self.dataset))))]
            #if data_random['category'] != self.dataset[index]['category']:
            if True:
                return data_random['rot_vec']