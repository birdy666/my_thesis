from math import nan
import random
import time
from torch.autograd import grad
import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter 
from utils import get_noise_tensor, print_performances, save_models

from torch.autograd import Variable


# algorithms: gan, wgan, wgan-gp, wgan-lp
# gan: k = 1, beta_1 = 0.5, beta_2 = 0.999, lr = 0.0001, epoch = 50~300
# wgan: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.001, c = 0.01, epoch = 200~1200
# wgan-gp: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.0004, lamb = 20, epoch = 200~1200
# wgan-lp: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.0004, lamb = 150, epoch = 200~1200
algorithm = 'wgan-gp'

# weight clipping (WGAN)
c = 0.01

def get_grad_penalty(batch_size, device, net_d, so3_real, so3_fake, text_match, text_match_mask, text_mismatch, text_mismatch_mask):  
    epsilon = torch.rand(batch_size, dtype=torch.float32).to(device)  
    ##########################
    # get so3_interpolated
    ##########################
    so3_interpolated = torch.empty_like(so3_real, dtype=torch.float32).to(device) 
    for j in range(batch_size):
        so3_interpolated[j] = epsilon[j] * so3_real[j] + (1 - epsilon[j]) * so3_fake[j]
    """# random sample
    epsilon = torch.rand(batch_size, 1, dtype=torch.float32)
    epsilon = epsilon.expand(batch_size, so3_real.nelement()//batch_size).contiguous().view(batch_size, so3_real.size()[-2], so3_real.size()[-1])
    
    # get so3_interpolated
    so3_interpolated  = epsilon * so3_real + ((1 - epsilon) * so3_fake)"""
    so3_interpolated = Variable(so3_interpolated, requires_grad=True)    
    # calculate gradient penalty
    score_interpolated_fake = net_d(so3_interpolated, text_match, text_match_mask)
    gradient_fake = grad(outputs=score_interpolated_fake, 
                    inputs=so3_interpolated, 
                    grad_outputs=torch.ones_like(score_interpolated_fake, requires_grad=False).to(device),
                    create_graph=True, 
                    retain_graph=True)[0]
    
    ##########################
    # get text_interpolated
    ##########################
    text_interpolated = torch.empty_like(text_match, dtype=torch.float32).to(device)  
    for j in range(batch_size):
        text_interpolated[j] = epsilon[j] * text_match[j] + (1 - epsilon[j]) * text_mismatch[j]    
    
    #text_interpolated = epsilon * text_match + ((1 - epsilon) * text_mismatch)
    text_interpolated = Variable(text_interpolated, requires_grad=True)

    f_mask = lambda x,y: torch.tensor([1 if x[i] or y[i] else 0 for i in range(len(x))])
    text_interpolated_mask = torch.empty_like(text_match_mask, dtype=torch.float32).to(device)
    text_interpolated_mask = Variable(text_interpolated_mask, requires_grad=True)
    for i in range(batch_size):
        text_interpolated_mask[i] = f_mask(text_match_mask[0], text_mismatch_mask[0])
    score_interpolated_wrong = net_d(so3_real, text_interpolated, text_interpolated_mask)
    gradient_wrong = grad(outputs=score_interpolated_wrong, 
                    inputs=text_interpolated, 
                    grad_outputs=torch.ones_like(score_interpolated_wrong).to(device),
                    create_graph=True, 
                    retain_graph=True)[0]

    """TODO 首先不確定把so3和text各跑一次net_d算梯度是否合理 再來不確定這裡算norm時dim=1正不正確"""    
    grad_fake_norm = gradient_fake.norm(2, dim=1)
    grad_wrong_norm = gradient_wrong.norm(2, dim=1)
    grad_penalty_fake = ((grad_fake_norm - 1) ** 2)
    grad_penalty_wrong = ((grad_wrong_norm - 1) ** 2)
    return grad_penalty_fake, grad_penalty_wrong

def get_d_loss(cfg, device, net_g, net_d, batch, optimizer_d=None, update_d=True):
    if update_d:
        net_d.zero_grad()

    # get so3, sentence vectors and noises
    so3_real = batch.get('so3').to(device) # torch.Size([128, 24, 3])
    text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
    text_match_mask = batch.get('vec_mask').to(device)
    text_mismatch = batch.get('vec_mismatch').to(device)
    text_mismatch_mask = batch.get('vec_mismatch_mask').to(device)
    noise = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)
    
    if update_d:
        # ground truth
        score_right = net_d(so3_real, text_match, text_match_mask).mean()
        # tex mismatch
        score_wrong = net_d(so3_real, text_mismatch, text_mismatch_mask).mean()
        # fake so3 by generator
        so3_fake = net_g(noise, text_match, text_match_mask).detach()
        score_fake = net_d(so3_fake, text_match, text_match_mask).mean()        
    else:
        with torch.no_grad():
            score_right = net_d(so3_real, text_match, text_match_mask).detach().mean()
            score_wrong = net_d(so3_real, text_mismatch, text_mismatch_mask).detach().mean()
            so3_fake = net_g(noise, text_match, text_match_mask).detach()
            score_fake = net_d(so3_fake, text_match, text_match_mask).detach().mean()

    if algorithm == 'wgan':
        if update_d:
            # calculate losses and update
            loss_d = cfg.SCORE_FAKE_WEIGHT_D*score_fake + cfg.SCORE_WRONG_WEIGHT_D*score_wrong - cfg.SCORE_RIGHT_WEIGHT_D*score_right
            loss_d.backward()
            optimizer_d.step_and_update_lr()
            # clipping
            for p in net_d.parameters():
                p.data.clamp_(-c, c)
    elif algorithm == 'wgan-gp':
        if update_d:
            grad_penalty_fake, grad_penalty_wrong = get_grad_penalty(cfg.BATCH_SIZE, device, net_d, so3_real, so3_fake, text_match, text_match_mask, text_mismatch, text_mismatch_mask)
            grad_penalty_fake = grad_penalty_fake.mean()
            grad_penalty_wrong = grad_penalty_wrong.mean()
        
            loss_d = cfg.SCORE_FAKE_WEIGHT_D * score_fake + cfg.SCORE_WRONG_WEIGHT_D * score_wrong \
                    - cfg.SCORE_RIGHT_WEIGHT_D * score_right \
                    +cfg.PENALTY_WEIGHT_FAKE * grad_penalty_fake + cfg.PENALTY_WEIGHT_WRONG * grad_penalty_wrong
                    #+ cfg.FAKE_WRONG_DIFF_WEIGHT * (((score_fake-score_wrong)**2)**0.5)
            loss_d.backward()
            optimizer_d.step_and_update_lr()
        else:
            grad_penalty_fake = 0
            grad_penalty_wrong = 0
    """else:
        # 'wgan-lp'
        loss_d = (score_fake + alpha * score_wrong - (1 + alpha) * score_right + lamb * (
            torch.max(torch.tensor(0, dtype=torch.float32, device=device), gradient_norm - 1).pow(2))).mean()
    """        
    return score_fake, score_wrong, score_right, grad_penalty_fake, grad_penalty_wrong

def get_g_loss(cfg, device, net_g, net_d, batch, optimizer_g=None, update_g=True):
    if update_g:
        net_g.zero_grad()

    # get sentence vectors and noises
    text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
    text_match_mask = batch.get('vec_mask').to(device)
    text_interpolated = batch.get('vec_interpolated').to(device)
    text_interpolated_mask = batch.get('vec_interpolated_mask').to(device)
    noise1 = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)
    noise2 = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)

    if update_g:
        # so3 fake
        so3_fake = net_g(noise1, text_match, text_match_mask)
        score_fake = net_d(so3_fake, text_match, text_match_mask).mean()
        # so3 interpolated
        so3_interpolated = net_g(noise2, text_interpolated, text_interpolated_mask)
        score_interpolated = net_d(so3_interpolated, text_interpolated, text_interpolated_mask).mean()
    else:
        with torch.no_grad():
            # so3 fake
            so3_fake = net_g(noise1, text_match, text_match_mask)
            score_fake = net_d(so3_fake, text_match, text_match_mask).detach().mean()
            # so3 interpolated
            so3_interpolated = net_g(noise2, text_interpolated, text_interpolated_mask)
            score_interpolated = net_d(so3_interpolated, text_interpolated, text_interpolated_mask).detach().mean()
    
    if update_g:
        # 'wgan', 'wgan-gp' and 'wgan-lp'
        loss_g = -(score_fake + score_interpolated)
        loss_g.backward()
        optimizer_g.step_and_update_lr()
    return score_fake, score_interpolated

def train_epoch(cfg, device, net_g, net_d, optimizer_g, optimizer_d, criterion, dataLoader_train, tb_writer=None, e=None):
    print('learning rate: g ' + str(optimizer_g._optimizer.param_groups[0].get('lr')) + ' d ' + str(optimizer_d._optimizer.param_groups[0].get('lr')))
    net_d.train()
    net_g.train()  
    total_loss_g = 0
    total_loss_d = 0

    for i, batch in enumerate(tqdm(dataLoader_train, desc='  - (Training)   ', leave=True)):        
        ###############################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###############################################################
        score_fake, score_wrong, score_right, grad_penalty_fake, grad_penalty_wrong = get_d_loss(cfg, device, net_g, net_d, batch, optimizer_d)         
        loss_d = cfg.SCORE_FAKE_WEIGHT_D * score_fake + cfg.SCORE_WRONG_WEIGHT_D * score_wrong \
                - cfg.SCORE_RIGHT_WEIGHT_D * score_right \
                +cfg.PENALTY_WEIGHT_FAKE * grad_penalty_fake + cfg.PENALTY_WEIGHT_WRONG * grad_penalty_wrong
        total_loss_d += loss_d.item()
        
        if tb_writer != None:
            tb_writer.add_scalars('loss_d_', {'score_fake': score_fake, 'score_wrong': score_wrong, 'score_right': score_right, 'grad_penalty_fake': grad_penalty_fake, 'grad_penalty_wrong': grad_penalty_wrong}, e*149+i)
        """# log
        writer.add_scalar('loss/d', loss_d, batch_number * (e - start_from_epoch) + i)"""
        ###############################################################
        # (2) Update G network: maximize log(D(G(z)))
        ###############################################################
        # after training discriminator for N times, train gernerator for 1 time
        if (i+1) % cfg.N_train_D_1_train_G == 0:
            # to avoid computation of net d
            for p in net_d.parameters():
                p.requires_grad = False
            #get losses
            score_fake, score_interpolated = get_g_loss(cfg, device, net_g, net_d, batch, optimizer_g)
            loss_g = -(score_fake + score_interpolated)
            total_loss_g += loss_g.item()
            # to enable computation of net d
            for p in net_d.parameters():
                p.requires_grad = True
            """# log
            writer.add_scalar('loss/g', loss_g, batch_number * (e - start_from_epoch) + i)"""
            if tb_writer != None:
                tb_writer.add_scalars('loss_g_', {'score_fake': score_fake, 'score_interpolated': score_interpolated}, e*149+i)
    return total_loss_g, total_loss_d

def val_epoch(cfg, device, net_g, net_d, criterion, dataLoader_val):
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

def train(cfg, device, net_g, net_d, optimizer_g, optimizer_d, criterion, dataLoader_train, dataLoader_val):   
    # tensorboard
    if cfg.USE_TENSORBOARD:
        print("[Info] Use Tensorboard")  
        tb_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tensorboard')) 
       
    # create log files
    log_train_file = os.path.join(cfg.OUTPUT_DIR, 'train.log')
    log_valid_file = os.path.join(cfg.OUTPUT_DIR, 'valid.log')
    print('[Info] Training performance will be written to file: {} and {}'.format(log_train_file, log_valid_file))
    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss_g,loss_d\n')
        log_vf.write('epoch,loss_g,loss_d\n')    
    start_of_all_training = time.time()
    for e in range(cfg.START_FROM_EPOCH, cfg.END_IN_EPOCH):   
        print("=====================Epoch " + str(e) + " start!=====================")     
        # Train!!
        start = time.time()
        train_loss_g, train_loss_d = train_epoch(cfg, device, net_g, net_d, optimizer_g, optimizer_d, criterion, dataLoader_train, tb_writer,e)
        
        lr_g=optimizer_g._optimizer.param_groups[0].get('lr')
        lr_d=optimizer_d._optimizer.param_groups[0].get('lr')
        print_performances('Training', start, train_loss_g, train_loss_d, lr_g, lr_d, e)
        
        # Validation!!
        start = time.time()
        val_loss_g, val_loss_d = val_epoch(cfg, device, net_g, net_d, criterion, dataLoader_val)
        print_performances('Validation', start, val_loss_g, val_loss_d, lr_g, lr_d, e)
        
        # save model for each 5 epochs
        if e % 5 == 4:
            save_models(cfg, e, net_g, net_d, optimizer_g.n_steps, optimizer_d.n_steps, cfg.CHKPT_PATH,  save_mode='all')
        elapse_mid=(time.time()-start_of_all_training)/60
        print('\n till episode ' + str(e) + ": " + str(elapse_mid) + " minutes (" + str(elapse_mid/60) + " hours)")

        """# Write log
        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss_g: 8.5f},{loss_d: 8.5f}\n'.format(
                epoch=e, loss_g=train_loss_g, loss_d=train_loss_d))
            log_vf.write('{epoch},{loss_g: 8.5f},{loss_d: 8.5f}\n'.format(
                epoch=e, loss_g=val_loss_g, loss_d=val_loss_d))"""

        if cfg.USE_TENSORBOARD:
            tb_writer.add_scalars('loss_g', {'train': train_loss_g, 'val': val_loss_g}, e)
            tb_writer.add_scalars('loss_d', {'train': train_loss_d, 'val': val_loss_d}, e)
            """tb_writer.add_scalar('learning_rate_g', lr_g, e)
            tb_writer.add_scalar('learning_rate_d', lr_d, e)"""
        

    elapse_final=(time.time()-start_of_all_training)/60
    print('\nfinished! ' + str(elapse_final) + " minutes")


if __name__ == "__main__":
    a = torch.tensor(0)

    print(a+1)

    
    