import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import random
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from utils_eval import one_nearest_neighbor, nearest_neighbor_distance, pose_distance
from data import getData
from config import cfg
from models.transformer import Decoder_G
from models.model_gan import Generator
from utils import get_noise_tensor
from tqdm import tqdm

device = torch.device('cpu')

_, dataset, dataset_val = getData(cfg, device)

checkpoint = torch.load('/media/remote_home/chang/z_master-thesis/models/checkpoints/epoch_204' + ".chkpt", map_location=torch.device('cpu')) #in docker

decoder_g = Decoder_G(n_layers=cfg.DEC_PARAM_G.n_layers, 
                            d_model=cfg.DEC_PARAM_G.d_model, 
                            d_inner_scale=cfg.DEC_PARAM_G.d_inner_scale, 
                            n_head=cfg.DEC_PARAM_G.n_head, 
                            d_k=cfg.DEC_PARAM_G.d_k, 
                            d_v=cfg.DEC_PARAM_G.d_v, 
                            dropout=cfg.DEC_PARAM_G.dropout, 
                            scale_emb=cfg.DEC_PARAM_G.scale_emb)    
    
                            
net_g = Generator(decoder_g, cfg.D_WORD_VEC).to(device)
net_g.load_state_dict(checkpoint['model_g'])
net_g.eval()

with torch.no_grad():
    real_poses = []
    fake_poses = []
    test_poses = []

    # classifier two-sample tests measure
    print("classifier two-sample tests measure")
    # two list of heatmaps: one real, one fake
    for i in tqdm(range(int(len(dataset.dataset))), desc=' - (two-sample tests)  ', leave=True):
        if i % 10 == 0:
            data = dataset.dataset[i]
            real_poses.append(torch.tensor(data['rot_vec'], dtype=torch.float32).to(device))
            sent_ix = random.randint(0, len(data['captions'])-1)
            caption = torch.tensor(data['captions'][sent_ix]).unsqueeze(0).to(device)
            mask = torch.tensor(data['caption_masks'][sent_ix]).unsqueeze(0).to(device)
            noise = get_noise_tensor(1, cfg.NOISE_SIZE).to(device)
            fake_poses.append(net_g(caption, mask, noise)[0].detach())

    # one-nearest-neighbor classification accuracy
    print('Classifier Two-sample Tests')
    print('accuracy: ' + str(one_nearest_neighbor(real_poses, fake_poses) * 100) + '%')

    """# image retrieval performance measure
    print("image retrieval performance measure")
    distance_real = []
    distance_fake = []

    # one test list of heatmaps
    for i in range(int(len(dataset_val.dataset))):
        data = dataset_val.dataset[i]
        test_poses.append(torch.tensor(data['rot_vec'], dtype=torch.float32).to(device))
    
    for i in tqdm(range(int(len(real_poses)*1)), desc=' - (IRPM)  ', leave=True):
        distance_real.append(nearest_neighbor_distance(real_poses[i], test_poses).item())
        distance_fake.append(nearest_neighbor_distance(fake_poses[i], test_poses).item())

    distance_real_fake = []
    distance_real_random = []
    distance_fake_random = []

    print("my test")
    for i in tqdm(range(int(len(real_poses)*1)), desc=' - (ccp)  ', leave=True):
        distance_real_fake.append(pose_distance(real_poses[i], fake_poses[i]).item())
        distance_real_random.append(pose_distance(real_poses[i],random.choice(test_poses)).item())
        distance_fake_random.append(pose_distance(fake_poses[i],random.choice(test_poses)).item())
        


    print('Image Retrieval Performance')
    print('mean nearest neighbor distance (real):' + str(np.mean(distance_real)))
    print('mean nearest neighbor distance (fake):' + str(np.mean(distance_fake)))

    # plot
    plt.figure()
    plt.hist(distance_real, np.linspace(0, 20, 100),  alpha=0.5)
    plt.hist(distance_fake, np.linspace(0, 20, 100),  alpha=0.5)
    plt.legend(['real', 'fake'])
    plt.title('nearest neighbor distances')
    plt.xlabel('distance')
    plt.ylabel('frequency')
    #plt.show()
    plt.savefig('Image_Retrieval_Performance.png')

    # plot
    plt.figure()
    plt.hist(distance_real_fake, np.linspace(0, 20, 40),  alpha=0.5)
    plt.hist(distance_real_random, np.linspace(0, 20, 40),  alpha=0.5)
    plt.hist(distance_fake_random, np.linspace(0, 20, 40),  alpha=0.5)
    plt.legend(['real_fake', 'real_random', 'fake_random'])
    plt.title('Sample distance')
    plt.xlabel('distance')
    plt.ylabel('frequency')
    #plt.show()
    plt.savefig('yoyodist')
    print('distance_real_fake:' + str(np.mean(distance_real_fake)))
    print('distance_real_random:' + str(np.mean(distance_real_random)))
    print('distance_fake_random:' + str(np.mean(distance_fake_random)))"""

    """print('Image Retrieval Performance')
    print('mean nearest neighbor distance (real):' + str(np.mean(distance_real)))
    print('mean nearest neighbor distance (fake):' + str(np.mean(distance_fake)))"""
    



