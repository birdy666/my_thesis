import fasttext
import fasttext.util
import numpy as np
from pycocotools.coco import COCO
import torch
import json
import os
import random
from tqdm import tqdm

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

def getEFTCaption(cfg, coco_caption):
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
        annids_captopn = coco_caption.getAnnIds()
        #annids_keypoints=coco_keypoints.getAnnIds()
        eft_all_with_caption = []
        for i in range(len(eft_data_all)):
            if eft_data_all[i]['annotId'] in annids_captopn:
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
    fasttext.util.reduce_model(text_model, 150)  
    eft_all_with_caption = getEFTCaption(cfg, coco_caption)    
    print("create dataloaders")
    # get the dataset (single person, with captions)
    train_size = int(len(eft_all_with_caption)*0.9)
    dataset_train = TheDataset(cfg, eft_all_with_caption[:train_size], coco_caption, text_model=text_model)
    dataset_val = TheDataset(cfg, eft_all_with_caption[train_size:], coco_caption, text_model=text_model)
    
    #return text_model, dataset, dataset_val, data_loader#, text_match_val, label_val
    return text_model, eft_all_with_caption, dataset_train, dataset_val


class TheDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, eft_all_with_caption, coco_caption, text_model=None, val=False):
        self.dataset = []
        self.cfg = cfg
        for i in range(len(eft_all_with_caption)):
            data = {'caption': coco_caption.loadAnns(eft_all_with_caption[i]['annotId'])[0]['caption'],
                    'parm_pose': eft_all_with_caption[i]['parm_pose'],
                    'parm_shape': eft_all_with_caption[i]['parm_shape'],
                    'smpltype': eft_all_with_caption[i]['smpltype'],
                    'annotId': eft_all_with_caption[i]['annotId'],
                    'imageName': eft_all_with_caption[i]['imageName']}
            data['so3'] = np.array([rotation2so3(R) for R in data['parm_pose']])
            # add sentence encoding
            if text_model is not None:
                # 加這一行資料量從19116 變 19083 但可以刷掉一些感覺就跟人沒有關的怪異資料
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
    print("This is data.py")
    """
    如果有朝一日想要換vector的為度
    ft = fasttext.load_model(cfg.TEXT_MODEL_PATH)
    print(ft.get_dimension())
    fasttext.util.reduce_model(ft, 100)
    print(ft.get_dimension())"""
    coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    #coco_caption_val = COCO(cfg.COCO_CAPTION_val)
    # load text encoding model
    text_model = fasttext.load_model(cfg.TEXT_MODEL_PATH)    
    eft_all_with_caption = getEFTCaption(cfg, coco_caption)    
    print("create dataloaders")
    # get the dataset (single person, with captions)
    train_size = int(len(eft_all_with_caption)*0.9)
    dataset_train = TheDataset(cfg, eft_all_with_caption[:train_size], coco_caption, text_model=text_model)
    dataLoader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.WORKERS)
    desc = '  - (Training)   '
    # torch.cat((noise_vector, sentence_vector), 1)
    # noise_tensor = torch.randn((number, noise_size, 1, 1), dtype=torch.float32)
    for i, batch in enumerate(tqdm(dataLoader_train, desc=desc)):
        """noise_tensor = torch.randn((number, 300, 1, 1), dtype=torch.float32)
        torch.cat((batch.get('vector'), noise_tensor), 1)"""
        print(batch.size()[0])
        
        break