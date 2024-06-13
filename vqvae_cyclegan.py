import torch
import torch.nn as nn
from vqvae.vqvae import VQVAE

##############################
##  残差块儿ResidualBlock
##############################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(                     ## block = [pad + conv + norm + relu + pad + conv + norm]
            nn.ReflectionPad2d(1),                      ## ReflectionPad2d():利用输入边界的反射来填充输入张量
            nn.Conv2d(in_features, in_features, 3),     ## 卷积
            nn.InstanceNorm2d(in_features),             ## InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
            nn.ReLU(inplace=True),                      ## 非线性激活
            nn.ReflectionPad2d(1),                      ## ReflectionPad2d():利用输入边界的反射来填充输入张量
            nn.Conv2d(in_features, in_features, 3),     ## 卷积
            nn.InstanceNorm2d(in_features),             ## InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
        )

    def forward(self, x):                               ## 输入为 一张图像
        return x + self.block(x)                        ## 输出为 图像加上网络的残差输出
    
class VQVAECYCLEGAN(VQVAE):
    def __init__(self, input_dim, dim, n_embedding, embedding: nn.Embedding):                            # dim是encoder后的编码向量的长度c
        super().__init__(input_dim, dim, n_embedding)
        self.vq_embedding = embedding                                               # 这里n_embedding就是码本里的向量个数K，也就是离散域的大小
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding,
                                               1.0 / n_embedding)                   # 使用均匀分布初始化码本
                                                                                    # 为什么要用embedding层做码本呢？因为在把离散index值转成连续的向量时，可以直接套这个层，在之后的decode里可以看到这个用法
                                                                                    
        channels = input_dim                                ## 输入通道数channels = 3

        ## 初始化网络结构
        out_features = int(dim/4)                                  ## 输出特征数out_features = 64 
        encoder = [                                           ## model = [Pad + Conv + Norm + ReLU]
            nn.ReflectionPad2d(channels),                   ## ReflectionPad2d(3):利用输入边界的反射来填充输入张量
            nn.Conv2d(channels, out_features, 7),           ## Conv2d(3, 64, 7)
            nn.InstanceNorm2d(out_features),                ## InstanceNorm2d(64):在图像像素上对HW做归一化，用在风格化迁移
            nn.ReLU(inplace=True),                          ## 非线性激活
        ]
        in_features = out_features                          ## in_features = 64

        ## 下采样，循环2次
        for _ in range(2):
            out_features *= 2                                                   ## out_features = 128 -> 256
            encoder += [                                                          ## (Conv + Norm + ReLU) * 2
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features                                          ## in_features = 256

        # 残差块儿，循环3次
        for _ in range(3):
            encoder += [ResidualBlock(out_features)]                              ## model += [pad + conv + norm + relu + pad + conv + norm]
        
        # 残差块儿，循环3次
        decoder = []
        for _ in range(3):
            decoder += [ResidualBlock(out_features)]

        # 上采样两次
        for _ in range(2):
            out_features //= 2                                                  ## out_features = 128 -> 64
            decoder += [                                                          ## model += [Upsample + conv + norm + relu]
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features                                          ## out_features = 64

        ## 网络输出层                                                            ## model += [pad + conv + tanh]
        decoder+= [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]    ## 将(3)的数据每一个都映射到[-1, 1]之间

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
    