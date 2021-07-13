from model import Generator2, Discriminator2, weights_init


from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad
import torch
import numpy as np

from heatmap import HeatmapDataset
from path import *
from utils import *



# whether multi-person
multi = False

# training parameters
batch_size = 128
learning_rate_g = 0.0004
learning_rate_d = 0.0004


# algorithms: gan, wgan, wgan-gp, wgan-lp
# gan: k = 1, beta_1 = 0.5, beta_2 = 0.999, lr = 0.0001, epoch = 50~300
# wgan: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.001, c = 0.01, epoch = 200~1200
# wgan-gp: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.0004, lamb = 20, epoch = 200~1200
# wgan-lp: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.0004, lamb = 150, epoch = 200~1200
algorithm = 'wgan-gp'

# weight clipping (WGAN)
c = 0.01

# penalty coefficient (Lipschitz Penalty or Gradient Penalty)
lamb = 10

# level of text-image matching
alpha = 1

# train discriminator k times before training generator
k = 5

# ADAM solver
beta_1 = 0.0
beta_2 = 0.9


def getLoss(net_g, net_d, criterion, dataset_val, heatmap_real_val, text_match_val, label_val):
    # validate
    net_g.eval()
    net_d.eval()

    # calculate d loss
    noise_val = get_noise_tensor(dataset_val.__len__()).to(device)
    text_mismatch_val = dataset_val.get_random_caption_tensor(dataset_val.__len__()).to(device)
    with torch.no_grad():
        score_right_val = net_d(heatmap_real_val, text_match_val).detach()
        score_wrong_val = net_d(heatmap_real_val, text_mismatch_val).detach()
        heatmap_fake_val = net_g(noise_val, text_match_val).detach()
        score_fake_val = net_d(heatmap_fake_val, text_match_val).detach()
    if algorithm == 'gan':
        label_val.fill_(1)
        loss_right_val = criterion(score_right_val.view(-1), label_val) * (1 + alpha)
        label_val.fill_(0)
        loss_fake_val = criterion(score_fake_val.view(-1), label_val)
        label_val.fill_(0)
        loss_wrong_val = criterion(score_wrong_val.view(-1), label_val) * alpha
        loss_d_val = loss_right_val + loss_fake_val + loss_wrong_val
    elif algorithm == 'wgan':
        loss_d_val = (score_fake_val + alpha * score_wrong_val - (1 + alpha) * score_right_val).mean()
    else:
        # 'wgan-gp' and 'wgan-lp'
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
            loss_d_val = (score_fake_val + alpha * score_wrong_val - (1 + alpha) * score_right_val + lamb * (
                torch.max(torch.tensor(0, dtype=torch.float32, device=device), gradient_norm_val - 1).pow(2))).mean()

    # calculate g loss
    text_interpolated_val = dataset_val.get_interpolated_caption_tensor(dataset_val.__len__()).to(device)
    noise_val = get_noise_tensor(dataset_val.__len__()).to(device)
    noise2_val = get_noise_tensor(dataset_val.__len__()).to(device)
    with torch.no_grad():
        heatmap_fake_val = net_g(noise_val, text_match_val).detach()
        heatmap_interpolated_val = net_g(noise2_val, text_interpolated_val).detach()
        score_fake_val = net_d(heatmap_fake_val, text_match_val).detach()
        score_interpolated_val = net_d(heatmap_interpolated_val, text_interpolated_val).detach()
    if algorithm == 'gan':
        label_val.fill_(1)
        loss_g_val = criterion(score_fake_val.view(-1), label_val) + criterion(score_interpolated_val.view(-1),
                                                                               label_val)
    else:
        # 'wgan', 'wgan-gp' and 'wgan-lp'
        loss_g_val = -(score_fake_val + score_interpolated_val).mean()
    
    return loss_g_val, loss_d_val

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

def train(net_g, net_d, optimizer_g, optimizer_d, criterion, fixedData):   
    
    saveModel_plotGenerative(net_g, net_d, fixedData, f'{start_from_epoch:05d}' + '_new', skeleton)
    # train
    start = datetime.now()
    print(start)
    print('training')
    net_g.train()
    net_d.train()
    iteration = 1
    writer = SummaryWriter(comment='_caption_' + ('multi_' if multi else '') + algorithm)
    loss_g = torch.tensor(0)
    loss_d = torch.tensor(0)

    # number of batches
    batch_number = len(data_loader)

    for e in range(start_from_epoch, end_in_epoch):
        print('learning rate: g ' + str(optimizer_g.param_groups[0].get('lr')) + ' d ' + str(
            optimizer_d.param_groups[0].get('lr')))

        for i, batch in enumerate(data_loader, 0):
            net_g.train()
            net_d.train()
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            net_d.zero_grad()

            # get heatmaps, sentence vectors and noises
            heatmap_real = batch.get('heatmap').to(device)
            text_match = batch.get('vector').to(device)
            current_batch_size = len(heatmap_real)
            text_mismatch = dataset.get_random_caption_tensor(current_batch_size).to(device)
            noise = get_noise_tensor(current_batch_size).to(device)

            """這裡和一般GAN不同的是他有三項，除了real, fake之外還多了一個wrong， 
            fake 是讓D能分辨出不合現實的pose，wrong是讓D能分辨出不對的描述"""
            # discriminate heatmpap-text pairs
            score_right = net_d(heatmap_real, text_match)
            score_wrong = net_d(heatmap_real, text_mismatch)

            # generate heatmaps
            heatmap_fake = net_g(noise, text_match).detach()

            # discriminate heatmpap-text pairs
            score_fake = net_d(heatmap_fake, text_match)

            if algorithm == 'gan':
                label = torch.full((current_batch_size,), 1, dtype=torch.float32, device=device)
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
                # 'wgan-gp' and 'wgan-lp'
                # random sample
                epsilon = np.random.rand(current_batch_size)
                heatmap_sample = torch.empty_like(heatmap_real)
                for j in range(current_batch_size):
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
                optimizer_d.step()

            # log
            writer.add_scalar('loss/d', loss_d, batch_number * (e - start_from_epoch) + i)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            if iteration == k:
                net_g.zero_grad()
                iteration = 0

                # get sentence vectors and noises
                text_interpolated = dataset.get_interpolated_caption_tensor(current_batch_size)
                noise = get_noise_tensor(current_batch_size)
                noise2 = get_noise_tensor(current_batch_size)
                text_interpolated = text_interpolated.to(device)
                noise = noise.to(device)
                noise2 = noise2.to(device)

                # generate heatmaps
                heatmap_fake = net_g(noise, text_match)
                heatmap_interpolated = net_g(noise2, text_interpolated)

                # discriminate heatmpap-text pairs
                score_fake = net_d(heatmap_fake, text_match)
                score_interpolated = net_d(heatmap_interpolated, text_interpolated)

                if algorithm == 'gan':
                    label = torch.full((current_batch_size,), 1, dtype=torch.float32, device=device)
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

                # log
                writer.add_scalar('loss/g', loss_g, batch_number * (e - start_from_epoch) + i)

            # print progress
            print('epoch ' + str(e + 1) + ' of ' + str(end_in_epoch) + ' batch ' + str(i + 1) + ' of ' + str(
                batch_number) + ' g loss: ' + str(loss_g.item()) + ' d loss: ' + str(loss_d.item()))

            iteration = iteration + 1


        saveModel_plotGenerative(net_g, net_d, fixedData, f'{e + 1:05d}', skeleton)        

        loss_g_val, loss_d_val = getLoss(net_g, net_d, criterion, dataset_val, heatmap_real_val, text_match_val, label_val)
      
        # print and log
        print('epoch ' + str(e + 1) + ' of ' + str(end_in_epoch) + ' val g loss: ' + str(
            loss_g_val.item()) + ' val d loss: ' + str(loss_d_val.item()))
        writer.add_scalar('loss_val/g', loss_g_val, (e - start_from_epoch))
        writer.add_scalar('loss_val/d', loss_d_val, (e - start_from_epoch))

        

    print('\nfinished')
    print(datetime.now())
    print('(started ' + str(start) + ')')
    writer.close()