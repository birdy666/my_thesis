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

def get_grad_penalty(batch_size, device, net_d, so3_real, so3_fake, text_match, text_match_mask):  
    epsilon = torch.rand(batch_size, dtype=torch.float32).to(device)  
    ##########################
    # get so3_interpolated
    ##########################
    fake_interpolated = torch.empty_like(so3_real, dtype=torch.float32).to(device) 
    for j in range(batch_size):
        fake_interpolated[j] = epsilon[j] * so3_real[j] + (1 - epsilon[j]) * so3_fake[j]
    
    fake_interpolated = Variable(fake_interpolated, requires_grad=True)    
    # calculate gradient penalty
    score_interpolated_fake = get_d_score(net_d(text_match, text_match_mask, fake_interpolated))
    gradient_fake = grad(outputs=score_interpolated_fake, 
                    inputs=fake_interpolated, 
                    grad_outputs=torch.ones_like(score_interpolated_fake).to(device),
                    create_graph=True, 
                    retain_graph=True)[0]
    grad_fake_norm = torch.sqrt(torch.sum(gradient_fake.reshape(batch_size, -1) ** 2, dim=1) + 1e-5)
    grad_penalty_fake = ((grad_fake_norm - 1) ** 2).mean()  
    return grad_penalty_fake

def get_d_score(so3_d):
    so3_d = so3_d.masked_fill(so3_d == 0, -1e9)
    pred = F.normalize(so3_d*so3_d, p=1, dim=-1)[:,:,:1]
    return pred.mean()

def get_g_so3(output):
    return output

def get_d_loss(cfg, device, net_g, net_d, batch, batch_index, optimizer_d=None, update_d=True):
    if update_d:
        net_d.zero_grad()    

    # get so3, text vectors and noises
    so3_real = batch.get('so3').to(device) # torch.Size([128, 24, 3])
    so3_wrong = batch.get('so3_wrong').to(device)
    text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
    text_match_mask = batch.get('vec_mask').to(device)
    noise = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)
    grad_penalty_fake = 0
    grad_penalty_wrong = 0
    score_fake = 0
    if not update_d:        
        with torch.no_grad():
            # ground truth
            score_right = net_d(text_match, text_match_mask, so3_real).detach()
            # so3 wrong
            score_wrong = net_d(text_match, text_match_mask, so3_wrong).detach()
            # fake so3 by generator
            so3_fake = get_g_so3(net_g(text_match, text_match_mask, noise).detach())
            score_fake = net_d(text_match, text_match_mask, so3_fake).detach()

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
            
            for _ in range(cfg.N_TRAIN_ENC):
                score_right = get_d_score(net_d(text_match, text_match_mask, so3_real))
                # so3 wrong
                score_wrong = get_d_score(net_d(text_match, text_match_mask, so3_wrong))             
            
                #grad_penalty_wrong = get_grad_penalty(cfg.BATCH_SIZE, device, net_d, so3_real, so3_wrong, text_match, text_match_mask)
                loss_d_rw = score_wrong - score_right
                loss_d_rw.backward()
                optimizer_d.step_and_update_lr()
                net_d.zero_grad()
            if batch_index % cfg.N_TRAIN_ENC_1_TRAIN_D == 0:
                net_d.zero_grad()
                """for p in net_d.encoder.parameters():
                    p.requires_grad=False"""
                score_right = get_d_score(net_d(text_match, text_match_mask, so3_real))
                # fake so3 by generator
                so3_fake = get_g_so3(net_g(text_match, text_match_mask, noise).detach())
                score_fake = get_d_score(net_d(text_match, text_match_mask, so3_fake)) 
                grad_penalty_fake = get_grad_penalty(cfg.BATCH_SIZE, device, net_d, so3_real, so3_fake, text_match, text_match_mask)
                loss_d_rf = score_fake  - score_right + cfg.PENALTY_WEIGHT_FAKE * grad_penalty_fake 
                loss_d_rf.backward()
                optimizer_d.step_and_update_lr()
                """for p in net_d.encoder.parameters():
                    p.requires_grad=True"""
        else:
            grad_penalty_fake = 0
            grad_penalty_wrong = 0
    if update_d:
        net_d.zero_grad()     
    return score_fake, score_wrong, score_right, grad_penalty_fake, grad_penalty_wrong

def get_g_loss(cfg, device, net_g, net_d, batch, optimizer_g=None, update_g=True):
    if update_g:
        net_g.zero_grad()

    # get text vectors and noises
    text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
    text_match_mask = batch.get('vec_mask').to(device)
    text_interpolated = batch.get('vec_interpolated').to(device)
    text_interpolated_mask = batch.get('vec_interpolated_mask').to(device)
    noise1 = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)
    noise2 = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)

    if update_g:
        # so3 fake
        output = net_g(text_match, text_match_mask, noise1)
        so3_fake = get_g_so3(output)
        score_fake = get_d_score(net_d(text_match, text_match_mask, so3_fake))
        # so3 interpolated
        so3_interpolated = get_g_so3(net_g(text_interpolated, text_interpolated_mask, noise2))
        score_interpolated = get_d_score(net_d(text_interpolated, text_interpolated_mask, so3_interpolated))
        # 'wgan', 'wgan-gp'
        loss_g = - (cfg.SCORE_FAKE_WEIGHT_G*score_fake + cfg.SCORE_INTERPOLATE_WEIGHT_G*score_interpolated)
        loss_g.backward()
        optimizer_g.step_and_update_lr()
    else:
        with torch.no_grad():            
            # so3 fake
            so3_fake = net_g(text_match, text_match_mask, noise1)
            score_fake = net_d(text_match, text_match_mask, so3_fake).detach()
            # so3 interpolated
            so3_interpolated = net_g(text_interpolated, text_interpolated_mask, noise2)
            score_interpolated = net_d(text_interpolated, text_interpolated_mask, so3_interpolated).detach()
    
    if update_g:
        net_g.zero_grad()
    return score_fake, score_interpolated

def train_epoch(cfg, device, net_g, net_d, optimizer_g, optimizer_d, dataLoader_train, tb_writer=None, e=None):
    print('learning rate: g ' + str(optimizer_g._optimizer.param_groups[0].get('lr')) + ' d ' + str(optimizer_d._optimizer.param_groups[0].get('lr')))
    net_d.train()
    net_g.train()  
    total_loss_g = 0
    total_loss_d = 0

    for i, batch in enumerate(tqdm(dataLoader_train, desc='  - (Training)   ', leave=False)):        
        ###############################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###############################################################
        score_fake, score_wrong, score_right, grad_penalty_fake, grad_penalty_wrong = get_d_loss(cfg, device, net_g, net_d, batch, i, optimizer_d)         
        loss_d = cfg.SCORE_FAKE_WEIGHT_D * score_fake + cfg.SCORE_WRONG_WEIGHT_D * score_wrong \
                    - cfg.SCORE_RIGHT_WEIGHT_D * score_right 
        if i % cfg.N_TRAIN_ENC ==0:
            total_loss_d += loss_d.item()
            if tb_writer != None:
                tb_writer.add_scalars('loss_d_', {'score_fake': score_fake, 'score_wrong': score_wrong, 'score_right': score_right, 'grad_penalty_fake': grad_penalty_fake, 'grad_penalty_wrong':grad_penalty_wrong}, e*len(dataLoader_train)+i)
                tb_writer.add_scalars('loss_d_wf', {'R_W': score_right-score_wrong, 'W_F': score_wrong-score_fake, 'w_loss': score_right-score_fake}, e*len(dataLoader_train)+i)
        
        ###############################################################
        # (2) Update G network: maximize log(D(G(z)))
        ###############################################################
        # after training discriminator for N times, train gernerator for 1 time
        if i % cfg.N_TRAIN_D_1_TRAIN_G == 0:
            """if cfg.SHARE_ENC:
                for p in net_g.encoder.parameters():
                    p.requires_grad=False  """   
            for _ in range(cfg.N_TRAIN_G):
                #get losses
                score_fake, score_interpolated= get_g_loss(cfg, device, net_g, net_d, batch, optimizer_g)
                loss_g =  - (cfg.SCORE_FAKE_WEIGHT_G*score_fake + cfg.SCORE_INTERPOLATE_WEIGHT_G*score_interpolated)
                total_loss_g += loss_g.item()
            """if cfg.SHARE_ENC:
                for p in net_g.encoder.parameters():
                    p.requires_grad=True   """        
            if tb_writer != None:
                tb_writer.add_scalars('loss_g_', {'score_fake': score_fake, 'score_interpolated': score_interpolated}, e*len(dataLoader_train)+i)
    return total_loss_g/ (cfg.BATCH_SIZE/cfg.N_TRAIN_D_1_TRAIN_G/cfg.N_TRAIN_G), total_loss_d/(cfg.BATCH_SIZE/cfg.N_TRAIN_ENC)

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
       
    start_of_all_training = time.time()
    for e in range(cfg.START_FROM_EPOCH, cfg.END_IN_EPOCH):   
        print("=====================Epoch " + str(e) + " start!=====================")     
        # Train!!
        start = time.time()
        train_loss_g, train_loss_d = train_epoch(cfg, device, net_g, net_d, optimizer_g, optimizer_d, dataLoader_train, tb_writer,e)
        
        lr_g=optimizer_g._optimizer.param_groups[0].get('lr')
        lr_d=optimizer_d._optimizer.param_groups[0].get('lr')
        print_performances('Training', start, train_loss_g, train_loss_d, lr_g, lr_d, e)
        
        # Validation!!
        """start = time.time()
        val_loss_g, val_loss_d = val_epoch(cfg, device, net_g, net_d, dataLoader_val)
        print_performances('Validation', start, val_loss_g, val_loss_d, lr_g, lr_d, e)"""
        
        # save model for each 5 epochs
        if e % cfg.SAVE_MODEL_ITR == 0 and e != 0:
            save_models(cfg, e, net_g, net_d, optimizer_g.n_steps, optimizer_d.n_steps, cfg.CHKPT_PATH,  save_mode='all')
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
    """from pycocotools.coco import COCO
    import json
    import string
    with open("/media/remote_home/chang/eft/eft_fit/COCO2014-All-ver01.json",'r') as f:
        eft_data = json.load(f)
        print("EFT data: ver {}".format(eft_data['ver']))
        eft_data_all = eft_data['data'] 
    coco_caption = COCO('/media/remote_home/chang/datasets/coco/annotations/captions_train2014.json')
    coco_keypoint = COCO('/media/remote_home/chang/datasets/coco/annotations/person_keypoints_train2014.json')
    
    sentences=[]
    img_iddd=[]
    previous_img_ids=[]
    for i in tqdm(range(len(eft_data_all)), desc='  - (Dataset)   ', leave=False):            
        # one eft data correspond to one keypoint in one img
        img_id = coco_keypoint.loadAnns(eft_data_all[i]['annotId'])[0]['image_id']
        if img_id in previous_img_ids:
            continue
        else:
            previous_img_ids.append(img_id)
            caption_ids = coco_caption.getAnnIds(imgIds=img_id)
            captions_anns = coco_caption.loadAnns(ids=caption_ids)
            caption_without_punctuation = ''.join([i for i in captions_anns[0]['caption'] if i not in string.punctuation])
            if len(caption_without_punctuation.split()) < 10:
                sentences.append(captions_anns[0]['caption'])
                img_iddd.append(img_id)
    with open('texthaha.txt', 'w') as f:
        for j, sentence in enumerate(sentences):
            line = str(img_iddd[j]) + ": " + sentence
            f.write(line)
            f.write('\n')"""
    mask = torch.tensor([[1,1,1,0],[1,1,0,0]])
    mask_b = mask.unsqueeze(-2).unsqueeze(1)
    a = torch.tensor([
                    [[1,1,1],
                    [2,2,2],
                    [3,3,3],
                    [4,4,4]],

                    [[5,5,5],
                    [6,6,6],
                    [7,7,7],
                    [8,8,8]]
                    ])
    print(a.unsqueeze(1).size())
    attn = torch.matmul(a.unsqueeze(1),a.unsqueeze(1).transpose(-2, -1))
    print(attn)
    print(attn.masked_fill(mask_b == 0, 0))