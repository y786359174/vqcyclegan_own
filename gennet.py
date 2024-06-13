import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import sys
sys.path.append('../')
from CycleGAN_own.network import weights_init_normal, GeneratorResNet, Discriminator

class Gen2(GeneratorResNet):
    def __init__(self, input_shape, num_residual_blocks, n_embedding):   ## (input_shape = (3, 256, 256), num_residual_blocks = 9)
        super().__init__(input_shape, num_residual_blocks, sample_n = 1, out_features = 32)
        self.n_embedding = n_embedding
        channels = input_shape[0]
        # self.out = nn.Conv2d(channels, n_embedding, 1)
        self.out = nn.Identity()
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x_fp = super().forward(x, is_noise=False)
        x_fp = torch.squeeze(x_fp, dim=1)
        x_quan = torch.clamp(torch.round(x_fp),min = 0, max = self.n_embedding-1).to(x_fp.device).detach()
        x = (x_fp + (x_quan - x_fp).detach())
        return x, x_fp, x_quan
            
##############################
#        Discriminator
##############################
class Dis2(Discriminator):
    def __init__(self, input_shape):                                        
        super().__init__(input_shape)

        channels, height, width = input_shape                                       ## input_shape:(3， 256， 256)

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)                  ## output_shape = (1, 16, 16)

        def discriminator_block(in_filters, out_filters, normalize=True):           ## 鉴别器块儿
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]   ## layer += [conv + norm + relu]    
            if normalize:                                                           ## 每次卷积尺寸会缩小一半，共卷积了4次
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(                                                 
            *discriminator_block(channels, 64, normalize=False),        ## layer += [conv(3, 64) + relu]
            *discriminator_block(64, 128),                              ## layer += [conv(64, 128) + norm + relu]
            *discriminator_block(128, 256),                             ## layer += [conv(128, 256) + norm + relu]
            # *discriminator_block(256, 512),                             ## layer += [conv(256, 512) + norm + relu]
            nn.ZeroPad2d((1, 0, 1, 0)),                                 ## layer += [pad]
            # nn.Conv2d(512, 512, 4, padding=1),                          ##
            nn.Conv2d(256, 1, 4, padding=1)                             ## layer += [conv(512, 1)]
        )
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = super().forward(x)
        return x