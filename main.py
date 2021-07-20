import torch
import torch.nn as nn
import numpy as np

from config import cfg
from train_conditional import train
from data import FixedData, getData
from model import getModels

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    text_model, eft_all_with_caption, dataset_train, dataset_val = getData(cfg)
    net_g, net_d = getModels(cfg, device, algorithm='wgan')
    #fixedData = FixedData(dataset_val, text_model, device)    

    # data loader, containing heatmap information
    dataLoader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.WORKERS)
    dataLoader_val = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.BATCH_SIZE, shuffle=False)
    """# data to validate
    data_val = enumerate(torch.utils.data.DataLoader(dataset_val, batch_size=dataset_val.__len__())).__next__()[1]
    text_match_val = data_val.get('vector').to(device)
    label_val = torch.full((len(dataset_val),), 1, dtype=torch.float32, device=device)"""

    optimizer_g = torch.optim.Adam(net_g.parameters(), lr=cfg.LEARNING_RATE_G, betas=(cfg.BETA_1, cfg.BETA_2))
    optimizer_d = torch.optim.Adam(net_d.parameters(), lr=cfg.LEARNING_RATE_D, betas=(cfg.BETA_1, cfg.BETA_2))
    criterion = nn.BCELoss()
    train(cfg, device, net_g, net_d, optimizer_g, optimizer_d, criterion, dataLoader_train, dataLoader_val)