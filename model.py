import torch
import torch.nn as nn
import torch.utils.data

from hyperparams import *
from path import *





def getModels(cfg, device, algorithm='wgan'):   
    print("create nn")

    net_g = Generator2(cfg).to(device)
    if algorithm == 'gan':
        net_d = Discriminator2(cfg, bn=True, sigmoid=True).to(device)
    elif algorithm == 'wgan':
        net_d = Discriminator2(cfg, bn=True).to(device)
    else:
        net_d = Discriminator2(cfg).to(device)
    net_g.apply(weights_init)
    net_d.apply(weights_init)

    """
    看有沒有之前沒訓練完的要接續
    """
    # load first step (without captions) trained weights if available
    if cfg.START_FROM_EPOCH > 0:
        net_g.load_state_dict(torch.load(cfg.GENERATOR_PATH + '/_' + f'{cfg.START_FROM_EPOCH:05d}'), False)
        net_d.load_state_dict(torch.load(cfg.DISCRIMINATOR_PATH + '/_' + f'{cfg.START_FROM_EPOCH:05d}'), False)
        net_g.first2.weight.data[0:cfg.NOISE_SIZE] = net_g.first.weight.data
        net_d.second2.weight.data[:, 0:cfg.convolution_channel_d[-1], :, :] = net_d.second.weight.data
    return net_g, net_d

# custom weights initialization called on net_g and net_d
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.normal_(m.bias.data, 0.0, 0.02)


# generator given noise input
class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()

        # several layers of transposed convolution, batch normalization and ReLu
        self.first = nn.ConvTranspose2d(cfg.NOISE_SIZE, cfg.CONVOLUTION_CHANNEL_G[0], 4, 1, 0, bias=False)
        self.main = nn.Sequential(
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_G[0]),
            nn.ReLU(True),

            nn.ConvTranspose2d(cfg.CONVOLUTION_CHANNEL_G[0], cfg.CONVOLUTION_CHANNEL_G[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_G[1]),
            nn.ReLU(True),

            nn.ConvTranspose2d(cfg.CONVOLUTION_CHANNEL_G[1], cfg.CONVOLUTION_CHANNEL_G[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_G[2]),
            nn.ReLU(True),

            nn.ConvTranspose2d(cfg.CONVOLUTION_CHANNEL_G[2], cfg.CONVOLUTION_CHANNEL_G[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_G[3]),
            nn.ReLU(True),

            
            nn.ConvTranspose2d(cfg.CONVOLUTION_CHANNEL_G[3], total_keypoints, 4, 2, 1, bias=False),
            nn.Tanh()

        )
        """TODO"""
    def forward(self, noise_vector):
        return self.main(self.first(noise_vector))

# generator given noise and text encoding input
class Generator2(Generator):
    def __init__(self, cfg):
        super(Generator2, self).__init__(cfg)
        g_input_size = cfg.NOISE_SIZE + cfg.COMPRESS_SIZE
        self.first2 = nn.ConvTranspose2d(g_input_size, cfg.CONVOLUTION_CHANNEL_G[0], 4, 1, 0, bias=False)
        self.cfg = cfg
        # compress text encoding first
        self.compress = nn.Sequential(
            nn.Linear(cfg.sentence_vector_size, cfg.COMPRESS_SIZE),
            nn.LeakyReLU(0.2, inplace=True)
        )
        """(sentence_vector_size, compress_size)=(30,128)"""

    def forward(self, noise_vector, sentence_vector):
        """
        noise_vector: 30, 128, 1, 1
        sentence_vector: 30, 300, 1, 1
        sentence_vector_size: 300
        compress_size: 128
        """
        # concatenate noise vector and compressed sentence vector
        input_vector = torch.cat((noise_vector, (
            (self.compress(sentence_vector.view(-1, self.cfg.sentence_vector_size))).view(-1, self.cfg.COMPRESS_SIZE, 1, 1))), 1)

        return self.main(self.first2(input_vector))


# discriminator given heatmap
class Discriminator(nn.Module):
    def __init__(self, cfg, bn=False, sigmoid=False):
        super(Discriminator, self).__init__()

        # several layers of convolution and leaky ReLu
        self.main = nn.Sequential(
            nn.Conv2d(cfg.total_keypoints, cfg.CONVOLUTION_CHANNEL_D[0], 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_D[0]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(cfg.CONVOLUTION_CHANNEL_D[0], cfg.CONVOLUTION_CHANNEL_D[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_D[1]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(cfg.CONVOLUTION_CHANNEL_D[1], cfg.CONVOLUTION_CHANNEL_D[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_D[2]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(cfg.CONVOLUTION_CHANNEL_D[2], cfg.CONVOLUTION_CHANNEL_D[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_D[3]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)

        )
        d_final_size = cfg.CONVOLUTION_CHANNEL_D[-1]
        self.second = nn.Conv2d(cfg.CONVOLUTION_CHANNEL_D[-1], d_final_size, 1, bias=False)
        self.third = nn.Sequential(
            nn.BatchNorm2d(d_final_size) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_final_size, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() if sigmoid else nn.Identity()

        )

    def forward(self, input_heatmap):
        return self.third(self.second(self.main(input_heatmap)))


# discriminator given heatmap and sentence vector
class Discriminator2(Discriminator):
    def __init__(self, cfg, bn=False, sigmoid=False):
        super(Discriminator2, self).__init__(cfg, bn, sigmoid)
        d_final_size = cfg.CONVOLUTION_CHANNEL_D[-1]
        # convolution with concatenated sentence vector
        self.second2 = nn.Conv2d(cfg.CONVOLUTION_CHANNEL_D[-1] + cfg.compress_size, d_final_size, 1, bias=False)

        # compress text encoding first
        self.compress = nn.Sequential(
            nn.Linear(cfg.sentence_vector_size, cfg.COMPRESS_SIZE),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.cfg = cfg

    def forward(self, input_heatmap, sentence_vector):
        # first convolution, then concatenate sentence vector
        tensor = torch.cat((self.main(input_heatmap), (
            (self.compress(sentence_vector.view(-1, self.cfg.sentence_vector_size))).view(-1, self.cfg.COMPRESS_SIZE, 1, 1)).repeat(1, 1,
                                                                                                                  4,
                                                                                                                  4)),
                           1)
        return self.third(self.second2(tensor))


