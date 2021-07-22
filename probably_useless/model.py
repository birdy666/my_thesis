import torch
import torch.nn as nn
import torch.utils.data

def getModels(cfg, device, algorithm='wgan'):   
    print("create nn")

    net_g = Generator(cfg).to(device)
    if algorithm == 'gan':
        net_d = Discriminator(cfg, bn=True, sigmoid=True).to(device)
    elif algorithm == 'wgan':
        net_d = Discriminator(cfg, bn=True).to(device)
    else:
        net_d = Discriminator(cfg).to(device)
    net_g.apply(weights_init)
    net_d.apply(weights_init)

    """
    看有沒有之前沒訓練完的要接續
    """
    # load first step (without captions) trained weights if available
    if cfg.START_FROM_EPOCH > 0:
        net_g.load_state_dict(torch.load(cfg.GENERATOR_PATH + '/_' + f'{cfg.START_FROM_EPOCH:05d}'), False)
        net_d.load_state_dict(torch.load(cfg.DISCRIMINATOR_PATH + '/_' + f'{cfg.START_FROM_EPOCH:05d}'), False)
        """WHYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY?????"""
        #self.first2 = nn.ConvTranspose2d(g_input_size, convolution_channel_g[0], 4, 1, 0, bias=False)
        net_g.first2.weight.data[0:cfg.NOISE_SIZE] = net_g.first.weight.data
        #self.second2 = nn.Conv2d(convolution_channel_d[-1] + compress_size, d_final_size, 1, bias=False)
        net_d.second2.weight.data[:, 0:cfg.CONVOLUTION_CHANNEL_D[-1], :, :] = net_d.second.weight.data
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




# generator given noise and text encoding input
class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.cfg = cfg        
        # compress text encoding first
        self.compress = nn.Sequential(
            nn.Linear(cfg.SENTENCE_VECTOR_SIZE, cfg.COMPRESS_SIZE),
            nn.LeakyReLU(0.2, inplace=True)
        )

        """(sentence_vector_size, compress_size)=(30,128)"""
        self.main = nn.Sequential(
            nn.ConvTranspose2d(cfg.NOISE_SIZE + cfg.COMPRESS_SIZE, cfg.CONVOLUTION_CHANNEL_G[0], 
                                kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_G[0]),
            nn.ReLU(True),

            nn.ConvTranspose2d(cfg.CONVOLUTION_CHANNEL_G[0], cfg.CONVOLUTION_CHANNEL_G[1], 
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_G[1]),
            nn.ReLU(True),

            nn.ConvTranspose2d(cfg.CONVOLUTION_CHANNEL_G[1], cfg.CONVOLUTION_CHANNEL_G[2], 
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_G[2]),
            nn.ReLU(True),

            nn.ConvTranspose2d(cfg.CONVOLUTION_CHANNEL_G[2], cfg.CONVOLUTION_CHANNEL_G[3], 
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_G[3]),
            nn.ReLU(True),

            
            nn.ConvTranspose2d(cfg.CONVOLUTION_CHANNEL_G[3], cfg.JOINT_NUM, 
                                kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()

        )

    def forward(self, noise_vector, sentence_vector):
        """
        noise_vector: 30, 128, 1, 1
        sentence_vector: 30, 300, 1, 1
        sentence_vector_size: 300
        compress_size: 128
        """
        # concatenate noise vector and compressed sentence vector
        first_half = noise_vector
        compressed_vector = self.compress(sentence_vector.view(-1, self.cfg.SENTENCE_VECTOR_SIZE))
        second_half = compressed_vector.view(-1, self.cfg.COMPRESS_SIZE, 1, 1)
        input_vector = torch.cat((first_half, second_half), 1)

        return self.main(input_vector)



# discriminator given heatmap and sentence vector
class Discriminator(nn.Module):
    def __init__(self, cfg, bn=False, sigmoid=False):
        super(Discriminator, self).__init__()  
        self.cfg = cfg      
        
        self.main = nn.Sequential(
            nn.Conv2d(cfg.JOINT_NUM, cfg.CONVOLUTION_CHANNEL_D[0], 
                        kernel_size=4, stride=2, padding=11, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_D[0]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(cfg.CONVOLUTION_CHANNEL_D[0], cfg.CONVOLUTION_CHANNEL_D[1], 
                        kernel_size=4, stride=2, padding=11, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_D[1]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(cfg.CONVOLUTION_CHANNEL_D[1], cfg.CONVOLUTION_CHANNEL_D[2], 
                        kernel_size=4, stride=2, padding=11, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_D[2]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(cfg.CONVOLUTION_CHANNEL_D[2], cfg.CONVOLUTION_CHANNEL_D[3], 
                        kernel_size=4, stride=2, padding=11, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_D[3]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # compress text encoding first
        self.compress = nn.Sequential(
            nn.Linear(cfg.SENTENCE_VECTOR_SIZE, cfg.COMPRESS_SIZE),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.last = nn.Sequential(
            # convolution with concatenated sentence vector
            nn.Conv2d(cfg.CONVOLUTION_CHANNEL_D[-1] + cfg.COMPRESS_SIZE, cfg.CONVOLUTION_CHANNEL_D[-1], 1, bias=False),
            nn.BatchNorm2d(cfg.CONVOLUTION_CHANNEL_D[-1]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cfg.CONVOLUTION_CHANNEL_D[-1], 1, 
                        kernel_size=4, stride=1, padding=10, bias=False),
            nn.Sigmoid() if sigmoid else nn.Identity()

        )

    def forward(self, input_heatmap, sentence_vector):
        # first convolution, then concatenate sentence vector
        # 256X4X4
        first_half = self.main(input_heatmap) 
        # 128x4x4
        compressed_vector = self.compress(sentence_vector.view(-1, self.cfg.SENTENCE_VECTOR_SIZE))
        second_half = compressed_vector.view(-1, self.cfg.COMPRESS_SIZE, 1, 1).repeat(1, 1, 4, 4)
        tensor = torch.cat((first_half, second_half),  dim=1)
        return self.last(tensor)

