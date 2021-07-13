import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim


from config import cfg
from train_conditional import train
from data import FixedData, getData
from model import getModels
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    text_model, data_loader = getData(cfg)
    net_g, net_d = getModels(cfg, device, algorithm='wgan')
    #fixedData = FixedData(dataset_val, text_model)    

    optimizer_g = optim.Adam(net_g.parameters(), lr=cfg.LEARNING_RATE_G, betas=(cfg.BETA_1, cfg.BETA_2))
    optimizer_d = optim.Adam(net_d.parameters(), lr=cfg.LEARNING_RATE_D, betas=(cfg.BETA_1, cfg.BETA_2))
    criterion = nn.BCELoss()
    train()