import fasttext
import numpy as np
from pycocotools.coco import COCO
import torch
import json
import os
import random

from geometry import rotation2so3
from utils import get_noise_tensor, get_caption_vector
from models.smpl import SMPL
from config import cfg

def getEFTCaption(cfg, coco_caption):
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
    if os.path.isfile(cfg.EFT_FIT_WITH_CAPTION_PATH):
        with open(cfg.EFT_FIT_WITH_CAPTION_PATH,'r') as f:
            eft_all_with_caption = json.load(f)
    else:
        print(f"Loading EFT data from {cfg.EFT_FIT_PATH}")
        with open(cfg.EFT_FIT_PATH,'r') as f:
            eft_data = json.load(f)
            print("EFT data: ver {}".format(eft_data['ver']))
        
            eft_data_all = eft_data['data']
        print(len(eft_data_all))    

        # Not all the annotID in eft_all are also in caption_train2014, need to filter them out
        annids = coco_caption.getAnnIds()
        eft_all_with_caption = []
        for i in range(len(eft_data_all)):
            if eft_data_all[i]['annotId'] in annids:
                eft_all_with_caption.append(eft_data_all[i])
        print(len(eft_all_with_caption))
        with open(cfg.EFT_FIT_WITH_CAPTION_PATH, 'w') as f:
            json.dump(eft_all_with_caption, f)
    
    return eft_all_with_caption

def getData(cfg):
    # read captions and keypoints from files
    coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    #coco_caption_val = COCO(cfg.COCO_CAPTION_val)
    # load text encoding model
    text_model = fasttext.load_model(cfg.TEXT_MODEL_PATH)
    
    eft_all_with_caption = getEFTCaption(cfg, coco_caption)
    """
    取得dataset
    """
    print("create dataloaders")
    # get the dataset (single person, with captions)
    train_size = int(len(eft_all_with_caption)*0.9)
    dataset_train = TheDataset(cfg, eft_all_with_caption[:train_size], coco_caption, text_model=text_model)
    dataset_val = TheDataset(cfg, eft_all_with_caption[train_size:], coco_caption, text_model=text_model)
    
    #return text_model, dataset, dataset_val, data_loader#, text_match_val, label_val
    return text_model, eft_all_with_caption, dataset_train, dataset_val



class FixedData():
    def __init__(self, dataset_val, text_model, device):
        # fixed training data (from validation set), noise and sentence vectors to see the progression
        self.h = 6
        self.w = 5
        self.train = dataset_val.get_random_heatmap_with_caption(self.w)
        fixed_real = self.train.get('heatmap').to(device)
        self.real_array = np.array(fixed_real.tolist()) * 0.5 + 0.5
        self.caption = self.train.get('caption')
        """6個128x1x1 ==> 4維陣列， 他是下面想要畫出6個假的pose"""
        self.noise = get_noise_tensor(self.h).to(device)
        self.text = torch.tensor([get_caption_vector(text_model, caption) for caption in self.caption], dtype=torch.float32,
                            device=device).unsqueeze(-1).unsqueeze(-1)

class TheDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, eft_all_with_caption, coco_caption, text_model=None, val=False):
        self.dataset = []
        self.cfg = cfg

        for i in range(len(eft_all_with_caption)):
            data = {'caption': coco_caption.loadAnns(eft_all_with_caption[i]['annotId'])[0]['caption'],
                    'parm_pose': eft_all_with_caption[i]['parm_pose'],
                    'parm_shape': eft_all_with_caption[i]['parm_shape'],
                    'smpltype': eft_all_with_caption[i]['smpltype']}
            data['so3'] = [rotation2so3(R) for R in data['parm_pose']]
            # add sentence encoding
            if text_model is not None:
                data['vector'] = get_caption_vector(text_model, data['caption'], cfg.ENCODING_WEIGHT)
                self.dataset.append(data)
      
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        item = dict()        
        # change heatmap range from [0,1] to[-1,1]
        item['parm_pose'] = torch.tensor(data['parm_pose'], dtype=torch.float32)

        item['vector'] = torch.tensor(data['vector'], dtype=torch.float32)
        """unsqueeze_ is in place operation, unsqueeze isn't"""
        item['vector'].unsqueeze_(-1).unsqueeze_(-1)
            
        return item

    # get a batch of random caption sentence vectors from the whole dataset
    def get_random_caption_tensor(self, number):
        vector_tensor = torch.empty((number, self.cfg.SENTENCE_VECTOR_SIZE), dtype=torch.float32)

        for i in range(number):
            # randomly select from all captions
            vector = random.choice(random.choice(self.dataset).get('vector'))
            vector_tensor[i] = torch.tensor(vector, dtype=torch.float32)
            
        return vector_tensor.unsqueeze_(-1).unsqueeze_(-1)
            

"""    # get a batch of random heatmaps and captions from the whole dataset
    def get_random_heatmap_with_caption(self, number):
        caption = []
        heatmap = torch.empty((number, total_keypoints, heatmap_size, heatmap_size), dtype=torch.float32)

        for i in range(number):
            # randomly select from all images
            data = random.choice(self.dataset)
            heatmap[i] = torch.tensor(self.get_heatmap(data, augment=False) * 2 - 1, dtype=torch.float32)
            caption.append(random.choice(data.get('caption')).get('caption'))

        return {'heatmap': heatmap, 'caption': caption}

    # get a batch of random interpolated caption sentence vectors from the whole dataset
    def get_interpolated_caption_tensor(self, batch_size):
        vector_tensor = torch.empty((batch_size, sentence_vector_size), dtype=torch.float32)

        for i in range(batch_size):
            # randomly select 2 captions from all captions
            vector = random.choice(random.choice(self.dataset).get('vector'))
            vector2 = random.choice(random.choice(self.dataset).get('vector'))

            # interpolate caption sentence vectors
            interpolated_vector = beta * vector + (1 - beta) * vector2
            vector_tensor[i] = torch.tensor(interpolated_vector, dtype=torch.float32)
        if self.for_regression:
            return vector_tensor
        else:
            return vector_tensor.unsqueeze_(-1).unsqueeze_(-1)
"""
if __name__ == "__main__":
    print("haha")

    x = {'a':1,'b':2}
    print(x.get('b'))
    # read captions and keypoints from files
    coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    #coco_caption_val = COCO(cfg.COCO_CAPTION_val)
    # load text encoding model
    #text_model = fasttext.load_model(cfg.TEXT_MODEL_PATH)
    
    eft_all_with_caption = getEFTCaption(cfg, coco_caption)
    print(eft_all_with_caption[0].keys())
    """dataset_train = TheDataset(cfg, eft_all_with_caption[:100], coco_caption, text_model=text_model)
    dataLoader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.WORKERS)
    for step, item in enumerate(dataLoader_train):
        print(step, item)
        break
    print(dataLoader_train)"""