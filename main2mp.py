import os
import sys
import time
import torch
import torch.nn as nn
from tqdm import tqdm
sys.path.append('../')
from CycleGAN_own.dataset import get_dataloader, get_img_shape
from xidataset import xiget_dataloader, xiget_img_shape
from gennet import Gen2, Dis2
from CycleGAN_own.utils import *
from vqvae.vqvae import VQVAE

import torch.distributed as dist ## DDP
# from torch.utils.data.distributed import DistributedSampler ## DDP
from torch.nn.parallel import DistributedDataParallel as DDP ## DDP

import cv2
import einops
import numpy as np

Distributed_Flag = True 
# torchrun -m --nnodes=1 --nproc_per_node=2 --master_port 29503 main2mp

def train(genGei:Gen2, genFei:Gen2, vqvaex:VQVAE, vqvaey:VQVAE,  disG:Dis2, disF:Dis2, n_embedding:int, ckpt_path, device = 'cuda'):
    print("using: ", device)    #你问我为什么写在这？因为main里总是有各种各样的地方会改device
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

    if Distributed_Flag:
        total_rank = dist.get_world_size()
        genGei = torch.nn.SyncBatchNorm.convert_sync_batchnorm(genGei).to(device)
        genGei = DDP(genGei, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        genFei = torch.nn.SyncBatchNorm.convert_sync_batchnorm(genFei).to(device)
        genFei = DDP(genFei, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        # vqvaex = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vqvaex).to(device)
        # vqvaex = DDP(vqvaex, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        # vqvaey = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vqvaey).to(device)
        # vqvaey = DDP(vqvaey, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        disG = torch.nn.SyncBatchNorm.convert_sync_batchnorm(disG).to(device)
        disG = DDP(disG, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        disF = torch.nn.SyncBatchNorm.convert_sync_batchnorm(disF).to(device)
        disF = DDP(disF, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    Imgx_dir_name = 'zqindex/selfish_faces64_xi'
    Imgy_dir_name = 'zqindex/anime_faces64_xi'
    
    dataloaderx = xiget_dataloader(batch_size,Imgx_dir_name, num_workers=0, distributed = Distributed_Flag)
    dataloadery = xiget_dataloader(batch_size,Imgy_dir_name, num_workers=0, distributed = Distributed_Flag)
    len_x = len(dataloaderx.dataset)
    len_y = len(dataloadery.dataset)

    if(len_x>len_y):
        len_dataset = len_y
    else:
        len_dataset = len_x
    

    # for param in vqvaex.parameters():
    #     param.requires_grad = False
    # for param in vqvaey.parameters():
    #     param.requires_grad = False
    
    gen_params = list(genGei.parameters()) + list(genFei.parameters())
    dis_params = list(disG.parameters()) + list(disF.parameters())

    dis_weight = 0.1
    optim_gen = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr, betas=(beta1, 0.999), weight_decay=0.0001)# betas 是一个优化参数，简单来说就是考虑近期的梯度信息的程度
    optim_dis = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                            dis_weight*lr, betas=(beta1, 0.999), weight_decay=0.0001)

    genGei = genGei.train()
    genFei = genFei.train()
    vqvaex = vqvaex.train()
    vqvaey = vqvaey.train()
    disG = disG.train()
    disF = disF.train()
    label_fake = torch.full((batch_size,1,2,2), 0.01, dtype=torch.float, device=device).detach()       # 真实图是1，虚假是0,需要注意,这里用的时候是计算loss，是小数，要用1. 和0.
    label_real = torch.full((batch_size,1,2,2), 0.99, dtype=torch.float, device=device).detach()       # 还有一件事，这里不能随意用batch_size，因为最后可能不满512，但是还是会继续算。得实时读取x的bn大小
                                                                    # 一般四个维度分别是bn c h w
                                                                    # 我把他从循环中挪出来并且检测x的bn不是batch_size就跳过
    xy_buffer = ReplayBuffer()
    yx_buffer = ReplayBuffer()

    k = 0
    for epoch_i in range(n_epochs):
        if Distributed_Flag and local_rank >= 0:
            torch.distributed.barrier()
            dataloaderx.sampler.set_epoch(epoch_i)
            dataloadery.sampler.set_epoch(epoch_i)
            
        tic = time.time()
        if(epoch_i%10==0 and k < k_max):          # 让k缓慢的增大到k_max
            k+=1
        # datax_iter = iter(dataloaderx)
        # datay_iter = iter(dataloadery)


        loss_list = torch.zeros((11,1)).to(device)
        loss_list = loss_list.squeeze()
        loader_i = 0
        # for _ in tqdm(range(len_dataset)):
        for xi, yi in tqdm(zip(dataloaderx, dataloadery)):
            loader_i = loader_i+1
            # x,_  = next(datax_iter)
            # y,_  = next(datay_iter)
            xi = xi.to(device)
            yi = yi.to(device)
            xi = torch.squeeze(xi, dim=1)
            yi = torch.squeeze(yi, dim=1)

            # 训练Gen
            for param in disG.parameters():
                param.requires_grad = False
            for param in disF.parameters():
                param.requires_grad = False
            # xi = vqvaex.encode(x).detach()
            # yi = vqvaey.encode(y).detach()
            xiGyi, xiGyi_fp, xiGyi_quan = genGei(xi.float())
            yiFxi, yiFxi_fp, yiFxi_quan = genFei(yi.float())
            # genG_commitment = mse_loss(xiGyi_fp, xiGyi_quan.detach())    
            # genF_commitment = mse_loss(yiFxi_fp, yiFxi_quan.detach())

            # xy = vqvaey.decode(xiGyi.long())
            # yx = vqvaex.decode(yiFxi.long())
            xiGyiFxi, _, _ = genFei(xiGyi)
            yiFxiGyi, _, _ = genGei(yiFxi)
            # g_loss_gan = torch.tensor(0.).to(device)
            
            g_loss_gan_G = criterion_GAN(disG(xiGyi.float()), label_real)
            g_loss_gan_F = criterion_GAN(disF(yiFxi.float()), label_real)
            g_loss_gan = g_loss_gan_G + g_loss_gan_F                      # woc xdm训练gen时千万记得别再傻呵呵用fake label当target了
            # g_loss_latent = criterion_Latent(xe.float(), xye.float()) + criterion_Latent(ye.float(), yxe.float())
            g_loss_cycle = criterion_Cycle(xiGyiFxi, xi.float()) + criterion_Cycle(yiFxiGyi, yi.float()) 
            g_loss = g_loss_gan*1 + g_loss_cycle*0.001
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

                xy_ = xy_buffer.push_and_pop(xiGyi).to(device).detach()
                yx_ = yx_buffer.push_and_pop(yiFxi).to(device).detach()
                dgr = disG(yi.float())
                dgf = disG(xy_)
                dfr = disF(xi.float())
                dff = disF(yx_)
                
                d_loss_G_real = criterion_GAN(dgr, label_real)
                d_loss_G_fake = criterion_GAN(dgf, label_fake)
                d_loss_F_real = criterion_GAN(dfr, label_real)
                d_loss_F_fake = criterion_GAN(dff, label_fake)
                d_loss = (d_loss_G_real + d_loss_G_fake + d_loss_F_real + d_loss_F_fake)  
                
                d_loss.backward()
                optim_dis.step()
            if(loader_i%10==0 and (Distributed_Flag==False or local_rank<=0)):
                with torch.no_grad():
                    x = vqvaex.decode(xi.long())
                    y = vqvaey.decode(yi.long())
                    xy = vqvaey.decode(xiGyi.long())
                    yx = vqvaex.decode(yiFxi.long())
                    # print(disG(xy), disF(yx))
                    # print(f'g_loss_gan_G {g_loss_gan_G.item():.4e} g_loss_gan_F {g_loss_gan_F.item():.4e}')
                    # print(f'd_loss_G_real {d_loss_G_real.item():.4e} d_loss_G_fake {d_loss_G_fake.item():.4e} d_loss_F_real {d_loss_F_real.item():.4e} d_loss_F_fake {d_loss_F_fake.item():.4e}')
                    x_stack = torch.stack((xy,yx,y,x), dim = 0)
                    x_new = einops.rearrange(x_stack, 'x n c h w -> (x h) (n w) c')
                    x_new = (x_new.clip(-1, 1) + 1) / 2 * 255
                    x_new = x_new.cpu().numpy().astype(np.uint8)
                    # print(x_new.shape)
                    if(x_new.shape[2]==3):
                        x_new = cv2.cvtColor(x_new, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(save_dir,'cycle_tmp.jpg'), x_new)
            
            loss_list += torch.stack([g_loss, d_loss, g_loss_gan, g_loss_cycle, torch.tensor(0.).to(device), torch.tensor(0.).to(device), torch.tensor(0.).to(device), d_loss_G_real, d_loss_G_fake, d_loss_F_real, d_loss_F_fake])  #把列表转换成tensor
        
        if Distributed_Flag and local_rank >= 0:
            torch.distributed.barrier()
        loss_list = loss_list/len_dataset    
        toc = time.time()    
        if Distributed_Flag==False or local_rank<=0:
            gan_weights = {'genGei': genGei.module.state_dict(), 'genFei': genFei.module.state_dict(), 'disG': disG.module.state_dict(), 'disF': disF.module.state_dict()}
            torch.save(gan_weights, ckpt_path)
            print(f'epoch {epoch_i} g_loss: {loss_list[0].item():.4e} d_loss: {loss_list[1].item():.4e} time: {(toc - tic):.2f}s')
            print(f'g_loss_gan {loss_list[2].item():.4e} g_loss_cycle: {loss_list[3].item():.4e} g_loss_identity: {loss_list[4].item():.4e} g_loss_latent: {loss_list[5].item():.4e} g_loss_recon: {loss_list[6].item():.4e}')
            print(f'd_loss_G_real {loss_list[7].item():.4e} d_loss_G_fake: {loss_list[8].item():.4e} d_loss_F_real: {loss_list[9].item():.4e} d_loss_F_fake: {loss_list[10].item():.4e}')
            if(epoch_i%1==0):
                sample(genGei.module, genFei.module, vqvaex, vqvaey, device=device)


sample_time = 0

def sample(genGei:Gen2, genFei:Gen2, vqvaex:VQVAE, vqvaey:VQVAE, device='cuda'):
    global sample_time
    sample_time += 1
    i_n = 10
    # for i in range(i_n*i_n):
    vqvaex = vqvaex.to(device)
    vqvaey = vqvaey.to(device)
    genGei = genGei.to(device)
    genGei = genGei.eval()
    genFei = genFei.to(device)
    genFei = genFei.eval()
    with torch.no_grad():
        Imgx_dir_name = '../face2face/dataset/train/C'
        Imgy_dir_name = '../face2face/dataset/train/A'
        dataloaderx = get_dataloader(i_n,Imgx_dir_name)
        dataloadery = get_dataloader(i_n,Imgy_dir_name)
        x,_  = next(iter(dataloaderx))
        y,_  = next(iter(dataloadery))
        x = x.to(device)
        y = y.to(device)
        xi = vqvaex.encode(x).float()
        yi = vqvaey.encode(y).float()
        xiGyi, _, _ = genGei(xi)
        yiFxi, _, _ = genFei(yi)
        xy = vqvaey.decode(xiGyi.long())
        yx = vqvaex.decode(yiFxi.long())
        xiGyiFxi, _, _ = genFei(xiGyi)
        yiFxiGyi, _, _ = genGei(yiFxi)
        xyx = vqvaex.decode(xiGyiFxi.long())
        yxy = vqvaey.decode(yiFxiGyi.long())
        x_stack = torch.stack((x, xy, xyx, y, yx, yxy), dim = 0)
        x_new = einops.rearrange(x_stack, 'x n c h w -> (x h) (n w) c')
        x_new = (x_new.clip(-1, 1) + 1) / 2 * 255
        x_new = x_new.cpu().numpy().astype(np.uint8)
        # print(x_new.shape)
        if(x_new.shape[2]==3):
            x_new = cv2.cvtColor(x_new, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir,'cyclegan_sample_%d.jpg' % (sample_time)), x_new)
    genGei = genGei.train()
    genFei = genFei.train()

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

save_dir = './data/face2anime64_2/'

if __name__ == '__main__':

    if Distributed_Flag:
        # setup
        local_rank = int(os.environ["LOCAL_RANK"]) ## DDP   
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        # 
        local_rank = dist.get_rank()
        total_rank = dist.get_world_size()
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        device = 'cuda'

    # random reset
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    ckpt_path = os.path.join(save_dir,'model_vqcyclegan.pth')
    # device = 'cuda'
    image_shape = get_img_shape()
    dim = 256
    n_embedding = 128
    vqvaex = VQVAE(image_shape[0], dim = dim, n_embedding = n_embedding).to(device)
    vqvaey = VQVAE(image_shape[0], dim = dim, n_embedding = n_embedding).to(device)
    n_embedding_d = xiget_img_shape()
    genGei = Gen2(n_embedding_d, 5, n_embedding).to(device)
    genFei = Gen2(n_embedding_d, 5, n_embedding).to(device)
    disG = Dis2(n_embedding_d).to(device)
    disF = Dis2(n_embedding_d).to(device)
    genGei.apply(weights_init_normal)
    genGei.apply(weights_init_normal)
    disG.apply(weights_init_normal)
    disF.apply(weights_init_normal)

    
    ckptx_path = 'zqindex/model_vqvae_selfish_faces64.pth'
    ckpty_path = 'zqindex/model_vqvae_anime_faces64.pth'
    vqvaex.load_state_dict(torch.load(ckptx_path))
    vqvaey.load_state_dict(torch.load(ckpty_path))

    # gan_weights = torch.load(ckpt_path)
    # genGei.load_state_dict(gan_weights['genGei'])
    # genFei.load_state_dict(gan_weights['genFei'])
    # disG.load_state_dict(gan_weights['disG'])
    # disF.load_state_dict(gan_weights['disF'])
    # sample(genGei, genFei, vqvaex, vqvaey, device=device)

    train(genGei, genFei, vqvaex, vqvaey, disG, disF ,n_embedding, ckpt_path, device=device)
    # gan_weights = torch.load(ckpt_path)
    # genGei.load_state_dict(gan_weights['genGei'])
    # genFei.load_state_dict(gan_weights['genFei'])
    # disG.load_state_dict(gan_weights['disG'])
    # disF.load_state_dict(gan_weights['disF'])

    # sample(genGei, genFei, vqvaex, vqvaey, device=device)
