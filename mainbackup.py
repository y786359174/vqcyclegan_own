import os
import sys
import time
import torch
import torch.nn as nn
from tqdm import tqdm
sys.path.append('../')
from CycleGAN_own.dataset import get_dataloader, get_img_shape
from CycleGAN_own.network import Discriminator
from CycleGAN_own.utils import *
from vqvae_cyclegan import VQVAECYCLEGAN

import cv2
import einops
import numpy as np

def train(vqvaex:VQVAECYCLEGAN, vqvaey:VQVAECYCLEGAN, disG:Discriminator, disF:Discriminator, embedding:nn.Embedding, ckpt_path, device = 'cuda'):
    
    n_epochs = 10000
    batch_size = 1
    lr = 1e-4
    beta1 = 0.5
    k_max = 6
    l_w_embedding=1
    l_w_commitment=0.25
    # criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_GAN = torch.nn.BCELoss().to(device)
    criterion_Cycle = torch.nn.MSELoss().to(device)
    criterion_Identity = torch.nn.MSELoss().to(device)
    criterion_Latent = torch.nn.L1Loss().to(device)
    criterion_Recon = torch.nn.MSELoss().to(device)
    mse_loss = nn.MSELoss().to(device)
    Imgx_dir_name = '../face2face/dataset/train/C'
    Imgy_dir_name = '../face2face/dataset/train/A'
    dataloaderx = get_dataloader(batch_size,Imgx_dir_name, num_workers=0)
    dataloadery = get_dataloader(batch_size,Imgy_dir_name, num_workers=0)
    len_x = len(dataloaderx.dataset)
    len_y = len(dataloadery.dataset)

    if(len_x>len_y):
        len_dataset = len_y
    else:
        len_dataset = len_x
    embed_param_names = [name for name, _ in embedding.named_parameters()]
    vqvaex_params = list(vqvaex.parameters())
    # vqvaey_params = list(vqvaey.parameters())
    gen_params = vqvaex_params + list(param for name, param in vqvaey.named_parameters() if name not in embed_param_names) 
    # gen_params = list(vqvaex.parameters()) + list(vqvaey.parameters())
    dis_params = list(disG.parameters()) + list(disF.parameters())

    dis_weight = 0.5
    optim_gen = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr, betas=(beta1, 0.999), weight_decay=0.0001)# betas 是一个优化参数，简单来说就是考虑近期的梯度信息的程度
    optim_dis = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                            dis_weight*lr, betas=(beta1, 0.999), weight_decay=0.0001)

    vqvaex = vqvaex.train()
    vqvaey = vqvaey.train()
    disG = disG.train()
    disF = disF.train()
    label_fake = torch.full((batch_size,1,3,3), 0.01, dtype=torch.float, device=device).detach()       # 真实图是1，虚假是0,需要注意,这里用的时候是计算loss，是小数，要用1. 和0.
    label_real = torch.full((batch_size,1,3,3), 0.99, dtype=torch.float, device=device).detach()       # 还有一件事，这里不能随意用batch_size，因为最后可能不满512，但是还是会继续算。得实时读取x的bn大小
                                                                    # 一般四个维度分别是bn c h w
                                                                    # 我把他从循环中挪出来并且检测x的bn不是batch_size就跳过
    xy_buffer = ReplayBuffer()
    yx_buffer = ReplayBuffer()

    k = 0
    for epoch_i in range(n_epochs):
        tic = time.time()
        if(epoch_i%50==0 and k < k_max):          # 让k缓慢的增大到k_max
            k+=1
        # datax_iter = iter(dataloaderx)
        # datay_iter = iter(dataloadery)


        loss_list = torch.zeros((11,1)).to(device)
        loss_list = loss_list.squeeze()

        # for _ in tqdm(range(len_dataset)):
        for (x, _), (y, _) in zip(dataloaderx, dataloadery):
            # x,_  = next(datax_iter)
            # y,_  = next(datay_iter)
            x = x.to(device)
            y = y.to(device)

            if(x.shape[0]!=batch_size):
                continue

            # 训练Gen
            
            for param in disG.parameters():
                param.requires_grad = False
            for param in disF.parameters():
                param.requires_grad = False
            
            x_hat, xze, xzq = vqvaex(x)
            l_reconstruct = mse_loss(x, x_hat)
            l_embedding = mse_loss(xze.detach(), xzq)     # 让zq去接近ze
            l_commitment = mse_loss(xze, xzq.detach())    # 让ze去接近zq
            g_loss_vqvaex = l_reconstruct + \
                l_w_embedding * l_embedding + l_w_commitment * l_commitment # 让zq多接近ze，码本多贴合真实数据

            y_hat, yze, yzq = vqvaey(y)
            l_reconstruct = mse_loss(y, y_hat)
            l_embedding = mse_loss(yze.detach(), yzq)     # 让zq去接近ze
            l_commitment = mse_loss(yze, yzq.detach())    # 让ze去接近zq
            g_loss_vqvaey = l_reconstruct + \
                l_w_embedding * l_embedding + l_w_commitment * l_commitment # 让zq多接近ze，码本多贴合真实数据

            g_loss_recon = g_loss_vqvaex + g_loss_vqvaey


            xe = vqvaex.encode_ze(x)
            xy = vqvaey.decode_ze(xe)
            ye = vqvaey.encode_ze(y)
            yx = vqvaex.decode_ze(ye)
            xye = vqvaey.encode_ze(xy)
            xyx = vqvaex.decode_ze(xye)
            yxe = vqvaex.encode_ze(yx)
            yxy = vqvaey.decode_ze(yxe)
            
            # g_loss_gan = torch.tensor(0.).to(device)
            
            
            g_loss_gan_G = criterion_GAN(disG(xy), label_real)
            g_loss_gan_F = criterion_GAN(disF(yx), label_real)
            g_loss_gan = g_loss_gan_G + g_loss_gan_F                      # woc xdm训练gen时千万记得别再傻呵呵用fake label当target了
            g_loss_latent = criterion_Latent(xe.float(), xye.float()) + criterion_Latent(ye.float(), yxe.float())
            g_loss_cycle = criterion_Cycle(xyx, x) + criterion_Cycle(yxy, y) 
            g_loss = g_loss_gan*20 + g_loss_cycle*1+ g_loss_latent*1 + g_loss_recon*10
            # g_loss = g_loss_recon*1000
            optim_gen.zero_grad()
            g_loss.backward()
            optim_gen.step()


            # 训练Dis
            if(epoch_i%k==0):
                for param in disG.parameters():
                    param.requires_grad = True
                for param in disF.parameters():
                    param.requires_grad = True
                optim_dis.zero_grad()

                xy_ = xy_buffer.push_and_pop(xy).to(device).detach()
                yx_ = yx_buffer.push_and_pop(yx).to(device).detach()
                dgr = disG(y)
                dgf = disG(xy_)
                dfr = disF(x)
                dff = disF(yx_)
                
                d_loss_G_real = criterion_GAN(dgr, label_real)
                d_loss_G_fake = criterion_GAN(dgf, label_fake)
                d_loss_F_real = criterion_GAN(dfr, label_real)
                d_loss_F_fake = criterion_GAN(dff, label_fake)
                d_loss = (d_loss_G_real + d_loss_G_fake + d_loss_F_real + d_loss_F_fake)  
                
                d_loss.backward()
                optim_dis.step()
            # with torch.no_grad():
            #     print(disG(xy), disF(yx))
            #     print(g_loss_gan_G, g_loss_gan_F)
            #     print(dgr, dgf, dfr, dff)
            #     x_stack = torch.stack((xy,yx,y,xy_,x,yx_), dim = 0)
            #     x_new = einops.rearrange(x_stack, 'x n c h w -> (x h) (n w) c')
            #     x_new = (x_new.clip(-1, 1) + 1) / 2 * 255
            #     x_new = x_new.cpu().numpy().astype(np.uint8)
            #     # print(x_new.shape)
            #     if(x_new.shape[2]==3):
            #         x_new = cv2.cvtColor(x_new, cv2.COLOR_RGB2BGR)
            #     cv2.imwrite(os.path.join(save_dir,'cycle_tmp.jpg'), x_new)

            # g_loss_latent = torch.tensor(0.).to(device)
            # g_loss_cycle = torch.tensor(0.).to(device)
            # g_loss_gan = torch.tensor(0.).to(device)
            # d_loss = torch.tensor(0.).to(device)
            # d_loss_G_real = torch.tensor(0.).to(device)
            # d_loss_G_fake = torch.tensor(0.).to(device)
            # d_loss_F_real = torch.tensor(0.).to(device)
            # d_loss_F_fake = torch.tensor(0.).to(device)
            loss_list += torch.stack([g_loss, d_loss, g_loss_gan, g_loss_cycle, torch.tensor(0.).to(device), g_loss_latent, g_loss_recon, d_loss_G_real, d_loss_G_fake, d_loss_F_real, d_loss_F_fake])  #把列表转换成tensor
            
        loss_list = loss_list/len_dataset    
        toc = time.time()    
        gan_weights = {'vqvaex': vqvaex.state_dict(), 'vqvaey': vqvaey.state_dict(), 'disG': disG.state_dict(), 'disF': disF.state_dict(), 'embed':embedding.state_dict()}
        torch.save(gan_weights, ckpt_path)
        print(f'epoch {epoch_i} g_loss: {loss_list[0].item():.4e} d_loss: {loss_list[1].item():.4e} time: {(toc - tic):.2f}s')
        print(f'g_loss_gan {loss_list[2].item():.4e} g_loss_cycle: {loss_list[3].item():.4e} g_loss_identity: {loss_list[4].item():.4e} g_loss_latent: {loss_list[5].item():.4e} g_loss_recon: {loss_list[6].item():.4e}')
        print(f'd_loss_G_real {loss_list[7].item():.4e} d_loss_G_fake: {loss_list[8].item():.4e} d_loss_F_real: {loss_list[9].item():.4e} d_loss_F_fake: {loss_list[10].item():.4e}')
        if(epoch_i%1==0):
            sample(vqvaex, vqvaey, device=device)

sample_time = 0

def sample(vqvaex:VQVAECYCLEGAN, vqvaey:VQVAECYCLEGAN, device='cuda'):
    global sample_time
    sample_time += 1
    i_n = 10
    # for i in range(i_n*i_n):
    vqvaex = vqvaex.to(device)
    vqvaex = vqvaex.eval()
    vqvaey = vqvaey.to(device)
    vqvaey = vqvaey.eval()
    with torch.no_grad():
        Imgx_dir_name = '../face2face/dataset/train/C'
        Imgy_dir_name = '../face2face/dataset/train/A'
        dataloaderx = get_dataloader(i_n,Imgx_dir_name)
        dataloadery = get_dataloader(i_n,Imgy_dir_name)
        x,_  = next(iter(dataloaderx))
        y,_  = next(iter(dataloadery))
        x = x.to(device)
        y = y.to(device)
        xe = vqvaex.encode_ze(x)
        x_hat = vqvaex.decode_ze(xe)
        xy = vqvaey.decode_ze(xe)
        ye = vqvaey.encode_ze(y)
        y_hat = vqvaey.decode_ze(ye)
        yx = vqvaex.decode_ze(ye)
        xye = vqvaey.encode_ze(xy)
        xyx = vqvaex.decode_ze(xye)
        yxe = vqvaex.encode_ze(yx)
        yxy = vqvaey.decode_ze(yxe)
        x_stack = torch.stack((x, xy, xyx, x_hat, y, yx, yxy, y_hat), dim = 0)
        x_new = einops.rearrange(x_stack, 'x n c h w -> (x h) (n w) c')
        x_new = (x_new.clip(-1, 1) + 1) / 2 * 255
        x_new = x_new.cpu().numpy().astype(np.uint8)
        # print(x_new.shape)
        if(x_new.shape[2]==3):
            x_new = cv2.cvtColor(x_new, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir,'cyclegan_sample_%d.jpg' % (sample_time)), x_new)
    vqvaex = vqvaex.train()
    vqvaey = vqvaey.train()

## 定义参数初始化函数
def weights_init_normal(m):                                    
    classname = m.__class__.__name__                        ## m作为一个形参，原则上可以传递很多的内容, 为了实现多实参传递，每一个moudle要给出自己的name. 所以这句话就是返回m的名字. 
    if classname.find("Conv") != -1:                        ## find():实现查找classname中是否含有Conv字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)     ## m.weight.data表示需要初始化的权重。nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        if hasattr(m, "bias") and m.bias is not None:       ## hasattr():用于判断m是否包含对应的属性bias, 以及bias属性是否不为空.
            torch.nn.init.constant_(m.bias.data, 0.0)       ## nn.init.constant_():表示将偏差定义为常量0.
    elif classname.find("BatchNorm2d") != -1:               ## find():实现查找classname中是否含有BatchNorm2d字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)     ## m.weight.data表示需要初始化的权重. nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        torch.nn.init.constant_(m.bias.data, 0.0)           ## nn.init.constant_():表示将偏差定义为常量0.

save_dir = './data/face2anime64/'

if __name__ == '__main__':

    ckpt_path = os.path.join(save_dir,'model_vqcyclegan.pth')
    device = 'cuda'
    image_shape = get_img_shape()
    # genG = GeneratorResNet(image_shape, 11).to(device)    # x->y
    # genF = GeneratorResNet(image_shape, 11).to(device)    # y->x
    dim = 256
    n_embedding = 128
    embedding = nn.Embedding(n_embedding, dim)
    vqvaex = VQVAECYCLEGAN(image_shape[0], dim = dim, n_embedding = n_embedding, embedding = embedding).to(device)
    vqvaey = VQVAECYCLEGAN(image_shape[0], dim = dim, n_embedding = n_embedding, embedding = embedding).to(device)
    # vqvaex = VQVAE(image_shape[0], dim = 256, n_embedding = 128).to(device)
    # vqvaey = VQVAE(image_shape[0], dim = 256, n_embedding = 128).to(device)
    disG = Discriminator(image_shape).to(device)
    disF = Discriminator(image_shape).to(device)
    vqvaex.apply(weights_init_normal)
    vqvaey.apply(weights_init_normal)
    disG.apply(weights_init_normal)
    disF.apply(weights_init_normal)
    # embedding.apply(weights_init_normal)  # 他有独特的初始化方法

    # gan_weights = torch.load(ckpt_path)
    # vqvaex.load_state_dict(gan_weights['vqvaex'])
    # vqvaex.load_state_dict(gan_weights['vqvaex'])
    # disG.load_state_dict(gan_weights['disG'])
    # disF.load_state_dict(gan_weights['disF'])
    # embedding.state_dict(gan_weights['embed'])
    # sample(vqvaex, vqvaex, device=device)

    train(vqvaex, vqvaey, disG, disF, embedding, ckpt_path, device=device)
    gan_weights = torch.load(ckpt_path)
    vqvaex.load_state_dict(gan_weights['vqvaex'])
    vqvaey.load_state_dict(gan_weights['vqvaey'])
    disG.load_state_dict(gan_weights['disG'])
    disF.load_state_dict(gan_weights['disF'])
    embedding.state_dict(gan_weights['embed'])

    sample(vqvaex, vqvaey, device=device)
