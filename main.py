import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from config import cfg
from train_conditional import train
from data import getData
from model_trans import getModels
from utils import get_noise_tensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    text_model, eft_all_with_caption, dataset_train, dataset_val = getData(cfg)
    net_g, net_d = getModels(cfg) 

    #dataLoader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.WORKERS)
    dataLoader_val = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    #optimizer_g = torch.optim.Adam(net_g.parameters(), lr=cfg.LEARNING_RATE_G, betas=(cfg.BETA_1, cfg.BETA_2))
    #optimizer_d = torch.optim.Adam(net_d.parameters(), lr=cfg.LEARNING_RATE_D, betas=(cfg.BETA_1, cfg.BETA_2))
    #criterion = nn.BCELoss()
    #train(cfg, device, net_g, net_d, optimizer_g, optimizer_d, criterion, dataLoader_train, dataLoader_val)

    print("YOOOOOOOOOOOOOOOOOOO")
    for i, batch in enumerate(tqdm(dataLoader_val, desc='  - (Validation)   ', leave=True)):
        net_g.eval()
        net_d.eval()
        so3_real = batch.get('so3').to(device) # torch.Size([128, 24, 3])
        text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
        text_match_mask = batch.get('vec_mask').to(device)
        noise1 = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)
        print(so3_real[0])
        with torch.no_grad():
            so3_fake = net_g(noise1, text_match, text_match_mask)
            print("嘻嘻")
            print(so3_fake.size())
            print(so3_fake[0])
            break
