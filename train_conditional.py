import random
import time
from torch.autograd import grad
import torch
import numpy as np
import os
from tqdm import tqdm
from time import sleep
from torch.utils.tensorboard import SummaryWriter 
from utils import get_noise_tensor


# algorithms: gan, wgan, wgan-gp, wgan-lp
# gan: k = 1, beta_1 = 0.5, beta_2 = 0.999, lr = 0.0001, epoch = 50~300
# wgan: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.001, c = 0.01, epoch = 200~1200
# wgan-gp: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.0004, lamb = 20, epoch = 200~1200
# wgan-lp: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.0004, lamb = 150, epoch = 200~1200
algorithm = 'wgan'

LOSS_SCALE = 10000000000

# weight clipping (WGAN)
c = 0.01

# penalty coefficient (Lipschitz Penalty or Gradient Penalty)
lamb = 10

# level of text-image matching
alpha = 1


def update_discriminator(cfg, device, net_g, net_d, optimizer_d, criterion, batch):
    net_d.train()
    loss_d = torch.tensor(0)
    net_d.zero_grad()

    # get so3, sentence vectors and noises
    so3_real = batch.get('so3').to(device) # torch.Size([128, 24, 3])
    text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
    text_match_mask = batch.get('vec_mask').to(device)
    text_mismatch = batch.get('vec_mismatch').to(device)
    text_mismatch_mask = batch.get('vec_mismatch_mask').to(device)
    noise = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)

    """這裡和一般GAN不同的是他有三項，除了real, fake之外還多了一個wrong， 
    fake 是讓D能分辨出不合現實的pose，wrong是讓D能分辨出不對的描述"""
    # discriminate heatmpap-text pairs
    score_right = net_d(so3_real, text_match, text_match_mask)
    score_wrong = net_d(so3_real, text_mismatch, text_mismatch_mask)

    # generate so3, 這裡要detach是因為我們是更新net_d不是net_g
    so3_fake = net_g(noise, text_match, text_match_mask)
    # discriminate so3-text pairs
    score_fake = net_d(so3_fake.detach(), text_match, text_match_mask)    
    if algorithm == 'gan':
        # torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
        label = torch.full((cfg.BATCH_SIZE,), 1, dtype=torch.float32, device=device)
        loss_right = criterion(score_right.view(-1), label) * (1 + alpha)
        loss_right.backward()

        label.fill_(0)
        loss_fake = criterion(score_fake.view(-1), label)
        loss_fake.backward()

        label.fill_(0)
        loss_wrong = criterion(score_wrong.view(-1), label) * alpha
        loss_wrong.backward()

        # calculate losses and update
        loss_d = loss_right + loss_fake + loss_wrong
        optimizer_d.step_and_update_lr()
    elif algorithm == 'wgan':
        # calculate losses and update
        loss_d = (score_fake + alpha * score_wrong - (1 + alpha) * score_right).mean()
        loss_d.backward()
        optimizer_d.step_and_update_lr()
        # clipping
        for p in net_d.parameters():
            p.data.clamp_(-c, c)
    else:
        print("先不測試ㄌ")
        """# 'wgan-gp' and 'wgan-lp'
        # random sample
        epsilon = np.random.rand(cfg.BATCH_SIZE)
        heatmap_sample = torch.empty_like(heatmap_real)
        for j in range(cfg.BATCH_SIZE):
            heatmap_sample[j] = epsilon[j] * heatmap_real[j] + (1 - epsilon[j]) * heatmap_fake[j]
        heatmap_sample.requires_grad = True
        text_match.requires_grad = True

        # calculate gradient penalty
        score_sample = net_d(heatmap_sample, text_match)
        gradient_h, gradient_t = grad(score_sample, [heatmap_sample, text_match], torch.ones_like(score_sample),
                                        create_graph=True)
        gradient_norm = (gradient_h.pow(2).sum((1, 2, 3)) + gradient_t.pow(2).sum((1, 2, 3))).sqrt()

        # calculate losses and update
        if algorithm == 'wgan-gp':
            loss_d = (score_fake + alpha * score_wrong - (1 + alpha) * score_right + lamb * (
                (gradient_norm - 1).pow(2))).mean()
        else:
            # 'wgan-lp'
            loss_d = (score_fake + alpha * score_wrong - (1 + alpha) * score_right + lamb * (
                torch.max(torch.tensor(0, dtype=torch.float32, device=device), gradient_norm - 1).pow(2))).mean()

        loss_d.backward()
        optimizer_d.step()"""
        
    return loss_d

def update_generator(cfg, device, net_g, net_d, optimizer_g, criterion, batch):
    net_g.train()
    loss_g = torch.tensor(0)
    net_g.zero_grad()

    # get sentence vectors and noises
    text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
    text_match_mask = batch.get('vec_mask').to(device)
    text_interpolated = batch.get('vec_interpolated').to(device)
    text_interpolated_mask = batch.get('vec_interpolated_mask').to(device)
    noise1 = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)
    noise2 = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)

    # generate so3
    so3_fake = net_g(noise1, text_match, text_match_mask)
    so3_interpolated = net_g(noise2, text_interpolated, text_interpolated_mask)

    # discriminate heatmpap-text pairs
    score_fake = net_d(so3_fake, text_match, text_match_mask)
    score_interpolated = net_d(so3_interpolated, text_interpolated, text_interpolated_mask)

    if algorithm == 'gan':
        label = torch.full((cfg.BATCH_SIZE,), 1, dtype=torch.float32, device=device)
        loss_g = criterion(score_fake.view(-1), label) + criterion(score_interpolated.view(-1), label)

        # calculate losses and update
        loss_g.backward()
        optimizer_g.step_and_update_lr()
    else:
        # 'wgan', 'wgan-gp' and 'wgan-lp'
        # calculate losses and update
        loss_g = -(score_fake + score_interpolated).mean()
        loss_g.backward()
        optimizer_g.step_and_update_lr()
    return loss_g

def train_epoch(cfg, device, net_g, net_d, optimizer_g, optimizer_d, criterion, dataLoader_train):
    print('learning rate: g ' + str(optimizer_g._optimizer.param_groups[0].get('lr')) + ' d ' + str(
            optimizer_d._optimizer.param_groups[0].get('lr')))
    iteration = 0    
    total_loss_g = 0
    total_loss_d = 0
    for i, batch in enumerate(tqdm(dataLoader_train, desc='  - (Training)   ', leave=True)):        
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        loss_d = update_discriminator(cfg, device, net_g, net_d, optimizer_d, criterion, batch)        
        total_loss_d += loss_d.item()
        
        """# log
        writer.add_scalar('loss/d', loss_d, batch_number * (e - start_from_epoch) + i)"""
        # after training discriminator for N times, train gernerator for 1 time
        if (i+1) % cfg.N_train_D_1_train_G == 0:
            # (2) Update G network: maximize log(D(G(z)))
            loss_g = update_generator(cfg, device, net_g, net_d, optimizer_g, criterion, batch)
            total_loss_g += loss_g.item()
            """# log
            writer.add_scalar('loss/g', loss_g, batch_number * (e - start_from_epoch) + i)"""

        iteration = iteration + 1

    return total_loss_g, total_loss_d

def get_d_loss(cfg, device, net_g, net_d, criterion, batch):
    # get so3, sentence vectors and noises
    so3_real = batch.get('so3').to(device) # torch.Size([128, 24, 3])
    text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
    text_match_mask = batch.get('vec_mask').to(device)
    text_mismatch = batch.get('vec_mismatch').to(device)
    text_mismatch_mask = batch.get('vec_mismatch_mask').to(device)
    # calculate d loss
    noise = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)
    with torch.no_grad():
        score_right = net_d(so3_real, text_match, text_match_mask).detach()
        score_wrong = net_d(so3_real, text_mismatch, text_mismatch_mask).detach()
        so3_fake = net_g(noise, text_match, text_match_mask).detach()
        score_fake = net_d(so3_fake, text_match, text_match_mask).detach()

    if algorithm == 'gan':
        label = torch.full((cfg.BATCH_SIZE,), 1, dtype=torch.float32, device=device)
        loss_right = criterion(score_right.view(-1), label) * (1 + alpha)
        label.fill_(0)
        loss_fake = criterion(score_fake.view(-1), label)
        label.fill_(0)
        loss_wrong = criterion(score_wrong.view(-1), label) * alpha
        loss_d = loss_right + loss_fake + loss_wrong
    elif algorithm == 'wgan':
        loss_d = (score_fake + alpha * score_wrong - (1 + alpha) * score_right).mean()
    else:
        print("先不測試ㄌ")
        """# 'wgan-gp' and 'wgan-lp'
        epsilon_val = np.random.rand(dataset_val.__len__())
        heatmap_sample_val = torch.empty_like(heatmap_real_val)
        for j in range(dataset_val.__len__()):
            heatmap_sample_val[j] = epsilon_val[j] * heatmap_real_val[j] + (1 - epsilon_val[j]) * heatmap_fake_val[j]
        heatmap_sample_val.requires_grad = True
        text_match_val.requires_grad = True
        score_sample_val = net_d(heatmap_sample_val, text_match_val)
        gradient_h_val, gradient_t_val = grad(score_sample_val, [heatmap_sample_val, text_match_val],
                                            torch.ones_like(score_sample_val), create_graph=True)
        gradient_norm_val = (gradient_h_val.pow(2).sum((1, 2, 3)) + gradient_t_val.pow(2).sum((1, 2, 3))).sqrt()
        if algorithm == 'wgan-gp':
            loss_d_val = (score_fake_val + alpha * score_wrong_val - (1 + alpha) * score_right_val + lamb * (
                (gradient_norm_val - 1).pow(2))).mean()

        else:
            # 'wgan-lp'
                oss_d_val = (score_fake_val + alpha * score_wrong_val - (1 + alpha) * score_right_val + lamb * (
                torch.max(torch.tensor(0, dtype=torch.float32, device=device), gradient_norm_val - 1).pow(2))).mean()
        """
    return loss_d

def get_g_loss(cfg, device, net_g, net_d, criterion, batch):    
    text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
    text_match_mask = batch.get('vec_mask').to(device)
    text_interpolated = batch.get('vec_interpolated').to(device)
    text_interpolated_mask = batch.get('vec_interpolated_mask').to(device)
    noise1 = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)
    noise2 = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)
        
    with torch.no_grad():
        # generate so3
        so3_fake = net_g(noise1, text_match, text_match_mask)
        so3__interpolated = net_g(noise2, text_interpolated, text_interpolated_mask)
            
        score_fake = net_d(so3_fake, text_match, text_match_mask).detach()
        score_interpolated = net_d(so3__interpolated, text_interpolated, text_interpolated_mask).detach()
    if algorithm == 'gan':
        label = torch.full((cfg.BATCH_SIZE,), 1, dtype=torch.float32, device=device)
        loss_g = criterion(score_fake.view(-1), label) + criterion(score_interpolated.view(-1), label)
    else:
        # 'wgan', 'wgan-gp' and 'wgan-lp'
        loss_g = -(score_fake + score_interpolated).mean()
    return loss_g

def val_epoch(cfg, device, net_g, net_d, criterion, dataLoader_val):
    # validate
    net_g.eval()
    net_d.eval()

    total_loss_g = 0
    total_loss_d = 0
    for i, batch in enumerate(tqdm(dataLoader_val, desc='  - (Validation)   ', leave=True)):
        # calculate d loss
        loss_d = get_d_loss(cfg, device, net_g, net_d, criterion, batch)
        total_loss_d += loss_d.item()

        # calculate g loss
        loss_g = get_g_loss(cfg, device, net_g, net_d, criterion, batch)
        total_loss_g += loss_g.item()

    return total_loss_g, total_loss_d   

def print_performances(header, start_time, loss_g, loss_d, lr_g, lr_d, e):
    print('  - {header:12} epoch {e}, loss_g: {loss_g: 8.5f}, loss_d: {loss_d:8.5f} %, lr_g: {lr_g:8.5f}, lr_d: {lr_d:8.5f}, '\
            'elapse: {elapse:3.3f} min'.format(
                e = e, header=f"({header})", loss_g=loss_g,
                loss_d=loss_d, elapse=(time.time()-start_time)/60, lr_g=lr_g, lr_d=lr_d))

def save_models(e, net_g, net_d, n_steps_g, n_steps_d, chkpt_path,  save_mode='all'):
    checkpoint = {'epoch': e, 'model_g': net_g.state_dict(), 'model_d': net_d.state_dict(),
                    'n_steps_g': n_steps_g, 'n_steps_d':n_steps_d}

    if save_mode == 'all':
        torch.save(checkpoint, chkpt_path + "/epoch_" + str(e) + ".chkpt")
    elif save_mode == 'best':
        pass
        """model_name = 'model.chkpt'
        if valid_loss <= min(valid_losses):
            torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
            print('    - [Info] The checkpoint file has been updated.')"""

def train(cfg, device, net_g, net_d, optimizer_g, optimizer_d, criterion, dataLoader_train, dataLoader_val):   
    # tensorboard
    if cfg.USE_TENSORBOARD:
        print("[Info] Use Tensorboard")  
        tb_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tensorboard'))  
        batch = next(iter(dataLoader_train))
        so3_real = batch.get('so3').to(device) # torch.Size([128, 24, 3])
        text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
        text_match_mask = batch.get('vec_mask').to(device)
        noise = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)
        tb_writer.add_graph(net_d, (so3_real, text_match, text_match_mask))
        tb_writer.add_graph(net_g, (noise, text_match, text_match_mask))
       
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
        train_loss_g, train_loss_d = train_epoch(cfg, device, net_g, net_d, optimizer_g, optimizer_d, criterion, dataLoader_train)
        lr_g=optimizer_g._optimizer.param_groups[0].get('lr')
        lr_d=optimizer_d._optimizer.param_groups[0].get('lr')
        print_performances('Training', start, train_loss_g, train_loss_d, lr_g, lr_d, e)
        

        # Evaluate!!
        start = time.time()
        val_loss_g, val_loss_d = val_epoch(cfg, device, net_g, net_d, criterion, dataLoader_val)
        print_performances('Validation', start, val_loss_g, val_loss_d, lr_g, lr_d, e)
        

        save_models(e, net_g, net_d, optimizer_g.n_steps, optimizer_d.n_steps, cfg.CHKPT_PATH,  save_mode='all')

        # Write log
        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss_g: 8.5f},{loss_d: 8.5f}\n'.format(
                epoch=e, loss_g=train_loss_g, loss_d=train_loss_d))
            log_vf.write('{epoch},{loss_g: 8.5f},{loss_d: 8.5f}\n'.format(
                epoch=e, loss_g=val_loss_g, loss_d=val_loss_d))

        if cfg.USE_TENSORBOARD:
            tb_writer.add_scalars('loss_g', {'train': train_loss_g, 'val': val_loss_g}, e)
            tb_writer.add_scalars('loss_d', {'train': train_loss_d, 'val': val_loss_d}, e)
            tb_writer.add_scalar('learning_rate_g', lr_g, e)
            tb_writer.add_scalar('learning_rate_d', lr_d, e)
        

    elapse=(time.time()-start_of_all_training)/60
    print('\nfinished! ' + str(elapse) + "minutes")


if __name__ == "__main__":
    print(5%6)
    print(12%6)
    print(7%6)