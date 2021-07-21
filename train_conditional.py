import time
from torch.autograd import grad
import torch
import numpy as np
import os
from tqdm import tqdm
from time import sleep

from utils import get_noise_tensor

# algorithms: gan, wgan, wgan-gp, wgan-lp
# gan: k = 1, beta_1 = 0.5, beta_2 = 0.999, lr = 0.0001, epoch = 50~300
# wgan: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.001, c = 0.01, epoch = 200~1200
# wgan-gp: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.0004, lamb = 20, epoch = 200~1200
# wgan-lp: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.0004, lamb = 150, epoch = 200~1200
algorithm = 'wgan'

# weight clipping (WGAN)
c = 0.01

# penalty coefficient (Lipschitz Penalty or Gradient Penalty)
lamb = 10

# level of text-image matching
alpha = 1


def saveModel_plotGenerative(net_g, net_d, fixedData, suffix, skeleton):
    # save models before training
    torch.save(net_g.state_dict(), generator_path + '_' + suffix)
    torch.save(net_d.state_dict(), discriminator_path + '_' + suffix)
    # plot and save generated samples from fixed noise (before training begins)
    net_g.eval()    
    with torch.no_grad():
        """
        每個row有5張對應到不同語句的假pose，用的都是同樣的noise。所以這裡用repeat_interleave(5, dim=0)
        每個text會場生6張假pose，所以用repeat(6, 1, 1, 1))
        """
        fixed_fake = net_g(fixedData.noise.repeat_interleave(fixedData.w, dim=0), fixedData.text.repeat(fixedData.h, 1, 1, 1))
    plot_generative_samples_from_noise(fixed_fake, fixedData.real_array, fixedData.caption, fixedData.w, fixedData.h, multi, skeleton, start_from_epoch)

def update_discriminator(cfg, device, net_g, net_d, optimizer_d, criterion, batch):
    net_d.train()
    loss_d = torch.tensor(0)
    net_d.zero_grad()

    # get so3, sentence vectors and noises
    so3_real = batch.get('so3').to(device) # torch.Size([128, 24, 3])
    text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
    text_match_mask = batch.get('vec_mask').to(device)
    text_mismatch = batch.get('vec_mismatch').to(device)
    noise = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)

    """這裡和一般GAN不同的是他有三項，除了real, fake之外還多了一個wrong， 
    fake 是讓D能分辨出不合現實的pose，wrong是讓D能分辨出不對的描述"""
    # discriminate heatmpap-text pairs
    score_right = net_d(so3_real, text_match)
    score_wrong = net_d(so3_real, text_mismatch)

    # generate so3, 這裡要detach是因為我們是更新net_d不是net_g
    so3_fake = net_g(noise, text_match, text_match_mask)
    # discriminate so3-text pairs
    score_fake = net_d(so3_fake.detach(), text_match)

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
        optimizer_d.step()
    elif algorithm == 'wgan':
        # calculate losses and update
        loss_d = (score_fake + alpha * score_wrong - (1 + alpha) * score_right).mean()
        loss_d.backward()
        optimizer_d.step()
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
    so3__interpolated = net_g(noise2, text_interpolated, text_interpolated_mask)

    # discriminate heatmpap-text pairs
    score_fake = net_d(so3_fake, text_match)
    score_interpolated = net_d(so3__interpolated, text_interpolated)

    if algorithm == 'gan':
        label = torch.full((cfg.BATCH_SIZE,), 1, dtype=torch.float32, device=device)
        loss_g = criterion(score_fake.view(-1), label) + criterion(score_interpolated.view(-1), label)

        # calculate losses and update
        loss_g.backward()
        optimizer_g.step()
    else:
        # 'wgan', 'wgan-gp' and 'wgan-lp'
        # calculate losses and update
        loss_g = -(score_fake + score_interpolated).mean()
        loss_g.backward()
        optimizer_g.step()
    return loss_g

def train_epoch(cfg, device, net_g, net_d, optimizer_g, optimizer_d, criterion, dataLoader_train):
    print('learning rate: g ' + str(optimizer_g.param_groups[0].get('lr')) + ' d ' + str(
            optimizer_d.param_groups[0].get('lr')))
    iteration = 1    
    total_loss_g = 0
    total_loss_d = 0
    for i, batch in enumerate(tqdm(dataLoader_train, desc='  - (Training)   ')):        
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        loss_d = update_discriminator(cfg, device, net_g, net_d, optimizer_d, criterion, batch)
        total_loss_d += loss_d
        # log
        writer.add_scalar('loss/d', loss_d, batch_number * (e - start_from_epoch) + i)

        # after training discriminator for N times, train gernerator for 1 time
        if iteration == cfg.N_train_D_1_train_G:
            # (2) Update G network: maximize log(D(G(z)))
            loss_g = update_generator()
            total_loss_g += loss_g
            # log
            writer.add_scalar('loss/g', loss_g, batch_number * (e - start_from_epoch) + i)

        # print progress
        print('epoch ' + str(e + 1) + ' of ' + str(end_in_epoch) + ' batch ' + str(i + 1) + ' of ' + str(
            batch_number) + ' g loss: ' + str(loss_g.item()) + ' d loss: ' + str(loss_d.item()))

        iteration = iteration + 1

    return total_loss_g, total_loss_d

def get_d_loss(cfg, device, net_g, net_d, criterion, batch):
    # get so3, sentence vectors and noises
    so3_real = batch.get('so3').to(device) # torch.Size([128, 24, 3])
    text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
    text_match_mask = batch.get('vec_mask').to(device)
    text_mismatch = batch.get('vec_mismatch').to(device)
    # calculate d loss
    noise = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)
    with torch.no_grad():
        score_right = net_d(so3_real, text_match).detach()
        score_wrong = net_d(so3_real, text_mismatch).detach()
        so3_fake = net_g(noise, text_match, text_match_mask).detach()
        score_fake = net_d(so3_fake, text_match).detach()

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
            
        score_fake = net_d(so3_fake, text_match).detach()
        score_interpolated = net_d(so3__interpolated, text_interpolated).detach()
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
    for i, batch in enumerate(tqdm(dataLoader_val, desc='  - (Validation)   ')):
        # calculate d loss
        loss_d = get_d_loss(cfg, device, net_g, net_d, criterion, batch)
        total_loss_d += loss_d

        # calculate g loss
        loss_g = get_g_loss(cfg, device, net_g, net_d, criterion, batch)
        total_loss_g += loss_g

    return total_loss_g, total_loss_d   

def print_performances(header, start_time, loss_g, loss_d, lr_g, lr_d):
    print('  - {header:12} loss_g: {loss_g: 8.5f}, loss_d: {loss_d:8.5f} %, lr_g: {lr_g:8.5f}, lr_d: {lr_d:8.5f}, '\
            'elapse: {elapse:3.3f} min'.format(
                header=f"({header})", loss_g=loss_g,
                loss_d=loss_d, elapse=(time.time()-start_time)/60, lr_g=lr_g, lr_d=lr_d))

def train(cfg, device, net_g, net_d, optimizer_g, optimizer_d, criterion, dataLoader_train, dataLoader_val):   
    # tensorboard
    if cfg.USE_TENSORBOARD:
        print("[Info] Use Tensorboard")    
        from torch.utils.tensorboard import SummaryWriter 
        tb_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tensorboard'))    
    # create log files
    log_train_file = os.path.join(cfg.OUTPUT_DIR, 'train.log')
    log_valid_file = os.path.join(cfg.OUTPUT_DIR, 'valid.log')
    print('[Info] Training performance will be written to file: {} and {}'.format(log_train_file, log_valid_file))
    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss_g,loss_d\n')
        log_vf.write('epoch,loss_g,loss_d\n')    

    for e in range(cfg.START_FROM_EPOCH, cfg.END_IN_EPOCH):        
        # Train!!
        start = time.time()
        train_loss_g, train_loss_d = train_epoch(cfg, device, net_g, net_d, optimizer_g, optimizer_d, criterion, dataLoader_train)
        lr_g=optimizer_g.param_groups[0].get('lr')
        lr_d=optimizer_d.param_groups[0].get('lr')
        print_performances('Training', start, train_loss_g, train_loss_d, lr_g, lr_d)

        # Evaluate!!
        start = time.time()
        val_loss_g, val_loss_d = val_epoch(cfg, device, net_g, net_d, criterion, dataLoader_val)
        print_performances('Validation', start, val_loss_g, val_loss_d, lr_g, lr_d)

        # Write log
        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss_g: 8.5f},{loss_d: 8.5f}\n'.format(
                epoch=e, loss_g=train_loss_g, loss_d=train_loss_d))
            log_vf.write('{epoch},{loss_g: 8.5f},{loss_d: 8.5f}\n'.format(
                epoch=e, loss_g=val_loss_g, loss_d=val_loss_d))

        if cfg.USE_TENSORBOARD:
            """tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_accu*100, 'val': valid_accu*100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)"""
            print("Haven't define what to draw man")
        

    print('\nfinished')


if __name__ == "__main__":
    index = 4
    others = list(range(0, index)) + list(range(index+1, 5))
    print(others)
    import random
    print(random.choice(others))