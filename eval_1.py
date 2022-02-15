import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import random
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from utils_eval import one_nearest_neighbor, nearest_neighbor_distance
from data import getData
from config import cfg
from models.transformer import Encoder, Decoder
from models.model_gan import Generator
from utils import get_noise_tensor
from tqdm import tqdm

device = torch.device('cpu')

_, dataset, dataset_val = getData(cfg, device)

checkpoint = torch.load('/media/remote_home/chang/z_master-thesis/models/checkpoints/epoch_36' + ".chkpt", map_location=torch.device('cpu')) #in docker
    
encoder_g = Encoder(n_layers=cfg.ENC_PARAM_G.n_layers, 
                            d_model=cfg.ENC_PARAM_G.d_model, 
                            d_inner_scale=cfg.ENC_PARAM_G.d_inner_scale, 
                            n_head=cfg.ENC_PARAM_G.n_head, 
                            d_k=cfg.ENC_PARAM_G.d_k, 
                            d_v=cfg.ENC_PARAM_G.d_v, 
                            dropout=cfg.ENC_PARAM_G.dropout, 
                            scale_emb=cfg.ENC_PARAM_G.scale_emb)
decoder_g = Decoder(n_layers=cfg.DEC_PARAM_G.n_layers, 
                            d_model=cfg.DEC_PARAM_G.d_model, 
                            d_inner_scale=cfg.DEC_PARAM_G.d_inner_scale, 
                            n_head=cfg.DEC_PARAM_G.n_head, 
                            d_k=cfg.DEC_PARAM_G.d_k, 
                            d_v=cfg.DEC_PARAM_G.d_v, 
                            dropout=cfg.DEC_PARAM_G.dropout, 
                            scale_emb=cfg.DEC_PARAM_G.scale_emb,G=True)    
  
                            
net_g = Generator(encoder_g,decoder_g, device, cfg.D_WORD_VEC, cfg.NOISE_WEIGHT_G).to(device)    
net_g.load_state_dict(checkpoint['model_g'])
net_g.eval()

with torch.no_grad():
    real_poses = []
    fake_poses = []
    test_poses = []

    # classifier two-sample tests measure

    # two list of heatmaps: one real, one fake
    for i in tqdm(range(len(dataset.dataset)), desc='  - (Dataset)   ', leave=True):
        data = dataset.dataset[i]
        real_poses.append(torch.tensor(data['rot_vec'], dtype=torch.float32))
        sent_ix = random.randint(0, len(data['captions'])-1)
        caption, mask = torch.tensor(data['captions'][sent_ix]).unsqueeze(0), torch.tensor(data['caption_masks'][sent_ix]).unsqueeze(0)
        noise = get_noise_tensor(1, cfg.NOISE_SIZE).to(device)
        fake_poses.append(net_g(caption, mask, noise)[0].detach())

    # one-nearest-neighbor classification accuracy
    print('Classifier Two-sample Tests')
    print('accuracy: ' + str(one_nearest_neighbor(real_poses, fake_poses) * 100) + '%')

    # image retrieval performance measure

    distance_real = []
    distance_fake = []

    # one test list of heatmaps
    for i in dataset_val.dataset:
        data = dataset.dataset[i]
        test_poses.append(torch.tensor(data['rot_vec'], dtype=torch.float32))

    for i in range(len(real_poses)):
        distance_real.append(nearest_neighbor_distance(real_poses[i], test_poses))
        distance_fake.append(nearest_neighbor_distance(fake_poses[i], test_poses))

    print('Image Retrieval Performance')
    print('mean nearest neighbor distance (real):' + str(np.mean(distance_real)))
    print('mean nearest neighbor distance (fake):' + str(np.mean(distance_fake)))

    # plot
    plt.figure()
    plt.hist(distance_real, np.arange(1,400,4), alpha=0.5)
    plt.hist(distance_fake, np.arange(1,400,4), alpha=0.5)
    plt.legend(['real', 'fake'])
    plt.title('nearest neighbor distances')
    plt.xlabel('distance')
    plt.ylabel('frequency')
    #plt.show()
    plt.savefig('Image_Retrieval_Performance.png')



