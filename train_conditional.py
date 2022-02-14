import torch
import torch.nn.functional as F
import math
import time
from torch.autograd import grad
import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter 
from utils import get_noise_tensor, print_performances, save_models
from torch.autograd import Variable

algorithm = 'wgan-gp'
# weight clipping (WGAN)
c = 1

def get_grad_penalty(batch_size, device, net_d, rot_vec_real, rot_vec_fake, caption, caption_mask):  
    epsilon = torch.rand(batch_size, dtype=torch.float32).to(device)  
    ##########################
    # get rot_vec_interpolated
    ##########################
    fake_interpolated = torch.empty_like(rot_vec_real, dtype=torch.float32).to(device) 
    for j in range(batch_size):
        fake_interpolated[j] = epsilon[j] * rot_vec_real[j] + (1 - epsilon[j]) * rot_vec_fake[j]
    interpolated = Variable(fake_interpolated, requires_grad=True)
    caption_emb =  Variable(net_d.embedding(caption), requires_grad=True)
    # calculate gradient penalty
    score_interpolated_fake = net_d(caption_emb, caption_mask, interpolated).mean()
    grads = grad(outputs=score_interpolated_fake, 
                    inputs=(interpolated, caption_emb), 
                    grad_outputs=torch.ones_like(score_interpolated_fake).to(device),
                    create_graph=True, 
                    retain_graph=True,
                    only_inputs=True)
    grad0 = grads[0].reshape(grads[0].size(0), -1)
    grad1 = grads[1].reshape(grads[1].size(0), -1)
    #grad01 = torch.cat((grad0,grad1),dim=1)        
    grad_fake_norm = torch.sqrt(torch.sum(grad0** 2, dim=1) + 1e-5)
    grad_penalty_fake = ((grad_fake_norm-1)**2).mean()

    grad_wrong_norm = torch.sqrt(torch.sum(grad1** 2, dim=1) + 1e-5)
    grad_penalty_wrong = ((grad_wrong_norm-1)**2).mean()

    return grad_penalty_fake, grad_penalty_wrong

def get_d_score(rot_vec_d):
    return rot_vec_d.mean()

def pad_text(text, d_word_vec):
    batch_s = text.size(0)
    new_text = torch.zeros((batch_s,24,d_word_vec), dtype=torch.float32)
    for i in range(batch_s):
        if len(text[i]) < 24:
            new_text[i][0:len(text[i])] = text[i]
            for j in range(len(text[i]), 24):
                new_text[i][j] = torch.zeros(d_word_vec, dtype=torch.float32).unsqueeze(0)
        
    return new_text

def get_d_loss(cfg, device, net_g, net_d, batch, batch_index, optimizer_d=None, update_d=True):
    if update_d:
        net_d.zero_grad()    

    # get rot_vec, text vectors and noises
    caption = batch.get('caption').to(device)
    rot_vec_real = batch.get('rot_vec').to(device) # torch.Size([128, 24, 3])
    rot_vec_wrong = batch.get('rot_vec_wrong').to(device)
    caption_mask = batch.get('caption_mask').to(device)
    
    noise = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)
    gp_fake = 0
    gp_wrong = 0
    score_fake = 0
    
    if update_d:
        # right
        caption_emb_r = net_d.embedding(caption)
        score_right = net_d(caption_emb_r, caption_mask, rot_vec_real).mean()
        # wrong
        caption_emb_w = net_d.embedding(caption)
        score_wrong = net_d(caption_emb_w, caption_mask, rot_vec_wrong).mean()
        # fake           
        rot_vec_fake = net_g(caption, caption_mask, noise)
        caption_emb_f = net_d.embedding(caption)
        score_fake = net_d(caption_emb_f, caption_mask, rot_vec_fake.detach()).mean() 
        # GP
        gp_fake, gp_wrong = get_grad_penalty(cfg.BATCH_SIZE, device, net_d, rot_vec_real, rot_vec_fake, caption, caption_mask)
        loss_d = score_wrong + score_fake - 2*score_right + 2*gp_fake + 2*gp_wrong
        
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()
        
    else:
        gp_fake = 0
        gp_wrong = 0

    if update_d:
        net_d.zero_grad()     
    return score_fake, score_wrong, score_right, gp_fake, gp_wrong

def get_g_loss(cfg, device, net_g, net_d, batch, optimizer_g=None, update_g=True):
    if update_g:
        net_g.zero_grad()
    # get rot_vec, text vectors and noises
    caption = batch.get('caption').to(device)
    caption_mask = batch.get('caption_mask').to(device)
    #caption, caption_mask, caption_len, rot_vec_real = prepare_data((caption, caption_mask, caption_len, rot_vec_real), device)
    
    # get text vectors and noises
    noise1 = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)

    if update_g:
        # rot_vec fake
        rot_vec_fake = net_g(caption, caption_mask, noise1)
        caption_emb_f = net_d.embedding(caption)
        score_fake = net_d(caption_emb_f, caption_mask, rot_vec_fake).mean()
        
        # 'wgan', 'wgan-gp'
        loss_g = - (score_fake)
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()
    else:
        with torch.no_grad():            
            # rot_vec fake
            rot_vec_fake = net_g(caption_emb, caption_mask, noise1)
            score_fake = net_d(caption_emb, caption_mask, rot_vec_fake).detach()
            
    
    if update_g:
        net_g.zero_grad()
    score_interpolated = 0
    return score_fake, score_interpolated

def train_epoch(cfg, device, net_g, net_d, optimizer_g, optimizer_d, dataLoader_train, tb_writer=None, e=None):
    net_d.train()
    net_g.train()  
    total_loss_g = 0
    total_loss_d = 0

    for i, batch in enumerate(tqdm(dataLoader_train, desc='  - (Training)   ', leave=False)):        
        ###############################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###############################################################
        
        score_fake, score_wrong, score_right, grad_penalty_fake, grad_penalty_wrong = get_d_loss(cfg, device, net_g, net_d, batch, i, optimizer_d)         
        loss_d = score_fake + score_wrong - 2*score_right  + grad_penalty_fake + grad_penalty_wrong
           
        total_loss_d += loss_d.item()
        if tb_writer != None:
            tb_writer.add_scalars('loss_d_', {'score_fake': score_fake, 'score_wrong': score_wrong, 'score_right': score_right, 'grad_penalty_fake': grad_penalty_fake, 'grad_penalty_wrong':grad_penalty_wrong}, e*len(dataLoader_train)+i)
            tb_writer.add_scalars('loss_d_wf', {'R_W': score_right-score_wrong, 'W_F': score_wrong-score_fake, 'w_loss': score_right-score_fake}, e*len(dataLoader_train)+i)
        
        ###############################################################
        # (2) Update G network: maximize log(D(G(z)))
        ###############################################################        
        if i % cfg.N_TRAIN_D == 0:
            score_fake, score_interpolated= get_g_loss(cfg, device, net_g, net_d, batch, optimizer_g)
            loss_g =  - (score_fake)
            total_loss_g += loss_g.item()
            if tb_writer != None:
                tb_writer.add_scalars('loss_g_', {'score_fake': score_fake, 'score_interpolated': score_interpolated}, e*len(dataLoader_train)+i)
    return total_loss_g/ (cfg.BATCH_SIZE), total_loss_d/(cfg.BATCH_SIZE*cfg.N_TRAIN_D)

def val_epoch(cfg, device, net_g, net_d, dataLoader_val):
    # validate
    net_g.eval()
    net_d.eval()

    total_loss_g = 0
    total_loss_d = 0
    for i, batch in enumerate(tqdm(dataLoader_val, desc='  - (Validation)   ', leave=True)):
        # calculate d loss
        score_fake, score_wrong, score_right, grad_penalty_fake, grad_penalty_wrong = get_d_loss(cfg, device, net_g, net_d, batch, update_d=False)
        loss_d = cfg.SCORE_FAKE_WEIGHT_D * score_fake + cfg.SCORE_WRONG_WEIGHT_D * score_wrong \
                    - cfg.SCORE_RIGHT_WEIGHT_D * score_right 
        total_loss_d += loss_d.item()

        # calculate g loss
        score_fake, score_interpolated = get_g_loss(cfg, device, net_g, net_d, batch, update_g=False)
        loss_g = -(score_fake + score_interpolated)
        total_loss_g += loss_g.item()
    return total_loss_g, total_loss_d   

def train(cfg, device, net_g, net_d, optimizer_g, optimizer_d, dataLoader_train, dataLoader_val):   
    # tensorboard
    if cfg.USE_TENSORBOARD:
        print("[Info] Use Tensorboard")  
        tb_writer = SummaryWriter(log_dir=cfg.TB_DIR) 

    """for p in net_g.embedding.parameters():
        p.requires_grad = False
    for q in net_d.embedding.parameters():
        q.requires_grad = False"""
       
    start_of_all_training = time.time()
    for e in range(cfg.START_FROM_EPOCH, cfg.END_IN_EPOCH):   
        print("=====================Epoch " + str(e) + " start!=====================")     
        # Train!!
        start = time.time()
        train_loss_g, train_loss_d = train_epoch(cfg, device, net_g, net_d, optimizer_g, optimizer_d, dataLoader_train, tb_writer, e)
        
        lr_g=cfg.LEARNING_RATE_G
        lr_d=cfg.LEARNING_RATE_D
        print_performances('Training', start, train_loss_g, train_loss_d, lr_g, lr_d, e)
        
        # Validation!!
        """start = time.time()
        val_loss_g, val_loss_d = val_epoch(cfg, device, net_g, net_d, dataLoader_val)
        print_performances('Validation', start, val_loss_g, val_loss_d, lr_g, lr_d, e)"""
        
        # save model for each 5 epochs
        if e % cfg.SAVE_MODEL_ITR == 0 and e != 0:
            save_models(cfg, e, net_g, net_d, cfg.CHKPT_PATH,  save_mode='all')
        elapse_mid=(time.time()-start_of_all_training)/60
        print('\n till episode ' + str(e) + ": " + str(elapse_mid) + " minutes (" + str(elapse_mid/60) + " hours)")

        if cfg.USE_TENSORBOARD:
            tb_writer.add_scalars('loss_g', {'train': train_loss_g/2, 'val': 0}, e)
            tb_writer.add_scalars('loss_d', {'train': train_loss_d, 'val': 0}, e)
            """tb_writer.add_scalar('loss_g', train_loss_g, e)
            tb_writer.add_scalar('loss_d', train_loss_d, e)"""
            """tb_writer.add_scalar('learning_rate_g', lr_g, e)
            tb_writer.add_scalar('learning_rate_d', lr_d, e)"""        

    elapse_final=(time.time()-start_of_all_training)/60
    print('\nfinished! ' + str(elapse_final) + " minutes")


if __name__ == "__main__":
    """a = torch.tensor([[[2,2,2,2,2]],[[2,2,2,2,2]]], dtype=torch.float32)
    print(a.size())
    a = a.unsqueeze(-2) # 1,5
    print(a.size())
    w = torch.ones(6, 5 , 4)

    print((a@w).size())"""

    a = torch.tensor([[[2,2,2,2,2],[5,5,5,5,5]],[[7,7,7,7,7],[8,8,8,8,8]]], dtype=torch.float32)
    print(a.size())
    print(a.view(2,-1)**2)