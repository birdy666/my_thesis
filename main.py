import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from optim import ScheduledOptim

from config import cfg
from train_conditional import train
from data import getData
from models.model_gan import getModels

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    checkpoint = None
    if cfg.START_FROM_EPOCH > 0:
        checkpoint = torch.load(cfg.CHKPT_PATH + "/epoch_" + str(cfg.START_FROM_EPOCH-1) + ".chkpt")
    text_model, eft_all_with_caption, dataset_train, dataset_val = getData(cfg)    
    net_g, net_d = getModels(cfg, device, checkpoint) 

    dataLoader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.WORKERS, drop_last=True)
    dataLoader_val = torch.utils.data.DataLoader(dataset_val, batch_size=cfg.BATCH_SIZE, shuffle=False, drop_last=True)
    
    # 這裡原本論文因為輸入輸出一樣 所以他這裡直接把d_model當作一個像是常數的參數 但我的模型不一樣 ...
    optimizer_g = ScheduledOptim(
        torch.optim.Adam(net_g.parameters(), lr=5e-4, betas=(cfg.BETA_1, cfg.BETA_2), eps=1e-09),
        lr_mul=2.0, d_model=300, n_warmup_steps=cfg.N_WARMUP_STEPS_G, n_steps=checkpoint['n_steps_g'] if checkpoint!=None else 0)
    optimizer_d = ScheduledOptim(
        torch.optim.Adam(net_d.parameters(), lr=1e-4, betas=(cfg.BETA_1, cfg.BETA_2), eps=1e-09),
        lr_mul=2.0, d_model=300, n_warmup_steps=cfg.N_WARMUP_STEPS_D, n_steps=checkpoint['n_steps_d'] if checkpoint!=None else 0)    
    criterion = nn.BCELoss()

    train(cfg, device, net_g, net_d, optimizer_g, optimizer_d, criterion, dataLoader_train, dataLoader_val)
    