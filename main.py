import torch
import torch.nn as nn
import numpy as np
from pycocotools.coco import COCO
import torch.optim as optim
import fasttext

from train_conditional import train
from utils import get_noise_tensor, get_caption_vector
from path import *
from utils import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

start_from_epoch = 0#200
end_in_epoch = 1#1200

class FixedData():
    def __init__(self, dataset_val, text_model):
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
      

def getData():
    """
    讀取caption和keypoint的annotation
    """
    # read captions and keypoints from files
    coco_caption = COCO(caption_path)
    coco_keypoint = COCO(keypoint_path)
    coco_caption_val = COCO(caption_path_val)
    coco_keypoint_val = COCO(keypoint_path_val)
    # keypoint connections (skeleton) from annotation file
    skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0].get('skeleton')) - 1
    # load text encoding model
    text_model = fasttext.load_model(text_model_path)

    """
    取得dataset
    """
    print("create dataloaders")
    # get the dataset (single person, with captions)
    dataset = HeatmapDataset(coco_keypoint, coco_caption, single_person=not multi, text_model=text_model, full_image=multi)
    dataset_val = HeatmapDataset(coco_keypoint_val, coco_caption_val, single_person=not multi, text_model=text_model,
                             full_image=multi)

    # data loader, containing heatmap information
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # data to validate
    data_val = enumerate(torch.utils.data.DataLoader(dataset_val, batch_size=dataset_val.__len__())).__next__()[1]
    text_match_val = data_val.get('vector').to(device)
    heatmap_real_val = data_val.get('heatmap').to(device)
    label_val = torch.full((len(dataset_val),), 1, dtype=torch.float32, device=device)

    return skeleton, text_model, dataset, dataset_val, data_loader, text_match_val, heatmap_real_val, label_val

def getModels(start_from_epoch):   
    print("create nn")

    net_g = Generator2().to(device)
    if algorithm == 'gan':
        net_d = Discriminator2(bn=True, sigmoid=True).to(device)
    elif algorithm == 'wgan':
        net_d = Discriminator2(bn=True).to(device)
    else:
        net_d = Discriminator2().to(device)
    net_g.apply(weights_init)
    net_d.apply(weights_init)

    """
    看有沒有之前沒訓練完的要接續
    """
    # load first step (without captions) trained weights if available
    if start_from_epoch > 0:
        net_g.load_state_dict(torch.load(generator_path + '_' + f'{start_from_epoch:05d}'), False)
        net_d.load_state_dict(torch.load(discriminator_path + '_' + f'{start_from_epoch:05d}'), False)
        net_g.first2.weight.data[0:noise_size] = net_g.first.weight.data
        net_d.second2.weight.data[:, 0:convolution_channel_d[-1], :, :] = net_d.second.weight.data
    return net_g, net_d

if __name__ == "__main__":
    skeleton, text_model, dataset, dataset_val, data_loader, text_match_val, heatmap_real_val, label_val = getData()
    net_g, net_d = getModels(start_from_epoch)

    

    fixedData = FixedData(dataset_val, text_model)
    

    optimizer_g = optim.Adam(net_g.parameters(), lr=learning_rate_g, betas=(beta_1, beta_2))
    optimizer_d = optim.Adam(net_d.parameters(), lr=learning_rate_d, betas=(beta_1, beta_2))
    criterion = nn.BCELoss()
    train()