import torch
import torch.nn as nn
import torch.utils.data

from hyperparams import *
from path import *

g_input_size = noise_size + compress_size
d_final_size = convolution_channel_d[-1]

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
    def __init__(self):
        super(Generator, self).__init__()

        # several layers of transposed convolution, batch normalization and ReLu
        self.first = nn.ConvTranspose2d(noise_size, convolution_channel_g[0], 4, 1, 0, bias=False)
        self.main = nn.Sequential(
            nn.BatchNorm2d(convolution_channel_g[0]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel_g[0], convolution_channel_g[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_g[1]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel_g[1], convolution_channel_g[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_g[2]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel_g[2], convolution_channel_g[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_g[3]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel_g[3], total_keypoints, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, noise_vector):
        return self.main(self.first(noise_vector))


# discriminator given heatmap
class Discriminator(nn.Module):
    def __init__(self, bn=False, sigmoid=False):
        super(Discriminator, self).__init__()

        # several layers of convolution and leaky ReLu
        self.main = nn.Sequential(
            nn.Conv2d(total_keypoints, convolution_channel_d[0], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_d[0]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel_d[0], convolution_channel_d[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_d[1]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel_d[1], convolution_channel_d[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_d[2]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel_d[2], convolution_channel_d[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_d[3]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)

        )
        self.second = nn.Conv2d(convolution_channel_d[-1], d_final_size, 1, bias=False)
        self.third = nn.Sequential(
            nn.BatchNorm2d(d_final_size) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_final_size, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() if sigmoid else nn.Identity()

        )

    def forward(self, input_heatmap):
        return self.third(self.second(self.main(input_heatmap)))


# generator given noise and text encoding input
class Generator2(Generator):
    def __init__(self):
        super(Generator2, self).__init__()

        self.first2 = nn.ConvTranspose2d(g_input_size, convolution_channel_g[0], 4, 1, 0, bias=False)

        # compress text encoding first
        self.compress = nn.Sequential(
            nn.Linear(sentence_vector_size, compress_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, noise_vector, sentence_vector):
        # concatenate noise vector and compressed sentence vector
        input_vector = torch.cat((noise_vector, (
            (self.compress(sentence_vector.view(-1, sentence_vector_size))).view(-1, compress_size, 1, 1))), 1)

        return self.main(self.first2(input_vector))


# discriminator given heatmap and sentence vector
class Discriminator2(Discriminator):
    def __init__(self, bn=False, sigmoid=False):
        super(Discriminator2, self).__init__(bn, sigmoid)

        # convolution with concatenated sentence vector
        self.second2 = nn.Conv2d(convolution_channel_d[-1] + compress_size, d_final_size, 1, bias=False)

        # compress text encoding first
        self.compress = nn.Sequential(
            nn.Linear(sentence_vector_size, compress_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input_heatmap, sentence_vector):
        # first convolution, then concatenate sentence vector
        tensor = torch.cat((self.main(input_heatmap), (
            (self.compress(sentence_vector.view(-1, sentence_vector_size))).view(-1, compress_size, 1, 1)).repeat(1, 1,
                                                                                                                  4,
                                                                                                                  4)),
                           1)
        return self.third(self.second2(tensor))


