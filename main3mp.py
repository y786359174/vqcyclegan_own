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
# from vqvae_cyclegan import VQVAECYCLEGAN
from vqvae.vqvae import VQVAE
from gennet import Gen3, Dis2, wgan_flag

import torch.distributed as dist ## DDP
# from torch.utils.data.distributed import DistributedSampler ## DDP
from torch.nn.parallel import DistributedDataParallel as DDP ## DDP

import cv2
import einops
import numpy as np

from torch.utils.tensorboard import SummaryWriter


Distributed_Flag = True 
# torchrun -m --nnodes=1 --nproc_per_node=2 --master_port 29504 main3mp

# def train(genGe: Gen3, genFe: Gen3, vqvaex:VQVAE, vqvaey:VQVAE, disG:Discriminator, disF:Discriminator, embedding:int, ckpt_path, device = 'cuda'):
def train(genGe: Gen3, genFe: Gen3, vqvaex:VQVAE, vqvaey:VQVAE, disG:Dis2, disF:Dis2, local_rank:int, ckpt_path, device = 'cuda'):
    print("using: ", device)    #你问我为什么写在这？因为main里总是有各种各样的地方会改device
    n_epochs = 10000
    batch_size = 2
    lr = 1e-5
    beta1 = 0.5
    k_max = 5
    l_w_embedding=1
    l_w_commitment=0.25
    # criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_GAN = torch.nn.BCELoss().to(device)
    criterion_Cycle = torch.nn.MSELoss().to(device)
    criterion_Identity = torch.nn.MSELoss().to(device)
    criterion_Latent = torch.nn.L1Loss().to(device)
    criterion_Recon = torch.nn.MSELoss().to(device)
    mse_loss = nn.MSELoss().to(device)

    # if Distributed_Flag:
    #     total_rank = dist.get_world_size()

    Imgx_dir_name = '../face2face/dataset/train/C'
    Imgy_dir_name = '../face2face/dataset/train/A'
    dataloaderx = get_dataloader(batch_size,Imgx_dir_name, num_workers=0, distributed = Distributed_Flag)
    dataloadery = get_dataloader(batch_size,Imgy_dir_name, num_workers=0, distributed = Distributed_Flag)
    len_x = len(dataloaderx.dataset)
    len_y = len(dataloadery.dataset)

    if(len_x>len_y):
        len_dataset = len_y
    else:
        len_dataset = len_x
    

    gen_params = list(genGe.parameters()) + list(genFe.parameters())
    dis_params = list(disG.parameters()) + list(disF.parameters())

    dis_weight = 0.1
    
    if not wgan_flag:
        optim_gen = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr, betas=(beta1, 0.999), weight_decay=0.0001)# betas 是一个优化参数，简单来说就是考虑近期的梯度信息的程度
        optim_dis = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                             dis_weight*lr, betas=(beta1, 0.999), weight_decay=0.0001)
    else:
        optim_gen = torch.optim.RMSprop([p for p in gen_params if p.requires_grad],lr) #
        optim_dis = torch.optim.RMSprop([p for p in dis_params if p.requires_grad],lr) #WGAN作者建议不要用Adam

    genGe = genGe.train()
    genFe = genFe.train()
    vqvaex = vqvaex.eval()
    vqvaey = vqvaey.eval()
    disG = disG.train()
    disF = disF.train()
    for param in vqvaex.parameters():
        param.requires_grad = False
    for param in vqvaey.parameters():
        param.requires_grad = False
    label_fake = torch.full((batch_size,1,1,1), 0., dtype=torch.float, device=device).detach()       # 真实图是1，虚假是0,需要注意,这里用的时候是计算loss，是小数，要用1. 和0.
    label_real = torch.full((batch_size,1,1,1), 1., dtype=torch.float, device=device).detach()       # 还有一件事，这里不能随意用batch_size，因为最后可能不满512，但是还是会继续算。得实时读取x的bn大小
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
        if(epoch_i%100==0 and k < k_max):          # 让k缓慢的增大到k_max
            k+=1
        # datax_iter = iter(dataloaderx)
        # datay_iter = iter(dataloadery)


        loss_list = torch.zeros((11,1)).to(device)
        loss_list = loss_list.squeeze()
        loader_i = 0
        # for _ in tqdm(range(len_dataset)):
        for (x, _), (y, _) in tqdm(zip(dataloaderx, dataloadery)):
            loader_i = loader_i+1
            x = x.to(device)
            y = y.to(device)
            xe = vqvaex.module.encoder(x) if Distributed_Flag else vqvaex.encoder(x)
            ye = vqvaey.module.encoder(y) if Distributed_Flag else vqvaey.encoder(y)
            xq = vqvaex.module.encode_ze2zq(xe) if Distributed_Flag else vqvaex.encode_ze2zq(xe)
            yq = vqvaey.module.encode_ze2zq(ye) if Distributed_Flag else vqvaey.encode_ze2zq(ye)
            xeGye = genGe(xe)
            yeFxe = genFe(ye)
            xy = vqvaey.module.decoder(xeGye) if Distributed_Flag else vqvaey.decoder(xeGye)
            yx = vqvaex.module.decoder(yeFxe) if Distributed_Flag else vqvaex.decoder(yeFxe)
            xeGyeFxe = genFe(xeGye)
            yeFxeGye = genGe(yeFxe)
            xyx = vqvaex.module.decoder(xeGyeFxe) if Distributed_Flag else vqvaex.decoder(xeGyeFxe)
            yxy = vqvaey.module.decoder(yeFxeGye) if Distributed_Flag else vqvaey.decoder(yeFxeGye)
            
            # 训练Dis
            if(loader_i%k==0):
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
                # xeGye_ = xy_buffer.push_and_pop(xeGye).to(device).detach()
                # yeFxe_ = yx_buffer.push_and_pop(yeFxe).to(device).detach()
                # dgr = disG(ye)
                # dgf = disG(xeGye_)
                # dfr = disF(xe)
                # dff = disF(yeFxe_)
                
                if not wgan_flag:
                    d_loss_G_real = criterion_GAN(dgr, label_real)
                    d_loss_G_fake = criterion_GAN(dgf, label_fake)
                    d_loss_F_real = criterion_GAN(dfr, label_real)
                    d_loss_F_fake = criterion_GAN(dff, label_fake)
                    d_loss = (d_loss_G_real + d_loss_G_fake + d_loss_F_real + d_loss_F_fake) 
                else:
                    d_loss_G_real = -torch.mean(dgr-dgf)
                    d_loss_F_real = -torch.mean(dfr-dff) 
                    d_loss =  d_loss_G_real+d_loss_F_real
                d_loss.backward()
                if Distributed_Flag==False or local_rank<=0:
                    Write_tensorboard('disG', list(disG.named_parameters()), epoch_i * len_dataset/2 + loader_i)# 记录梯度信息
                    Write_tensorboard('disF', list(disF.named_parameters()), epoch_i * len_dataset/2 + loader_i)# 记录梯度信息
                optim_dis.step()
                if not wgan_flag:
                    pass
                else:
                    for p in dis_params:
                        p.data.clamp_(-0.01, 0.01)
            # 训练Gen
            for param in disG.parameters():
                param.requires_grad = False
            for param in disF.parameters():
                param.requires_grad = False
            # g_loss_gan = torch.tensor(0.).to(device)

            xe_commitment = mse_loss(xe, xq.detach())
            ye_commitment = mse_loss(ye, yq.detach())
            g_loss_commitment = xe_commitment + ye_commitment
            if not wgan_flag:
                g_loss_gan_G = criterion_GAN(disG(xy), label_real)
                g_loss_gan_F = criterion_GAN(disF(yx), label_real)
                # g_loss_gan_G = criterion_GAN(disG(xeGye), label_real)
                # g_loss_gan_F = criterion_GAN(disF(yeFxe), label_real)
                g_loss_gan = g_loss_gan_G + g_loss_gan_F                      # woc xdm训练gen时千万记得别再傻呵呵用fake label当target了
            else:
                g_loss_gan_G = -torch.mean(disG(xy))
                g_loss_gan_F = -torch.mean(disF(yx)) 
                # g_loss_gan_G = -torch.mean(disG(xeGye))
                # g_loss_gan_F = -torch.mean(disF(yeFxe)) 
                g_loss_gan = g_loss_gan_G + g_loss_gan_F
            # g_loss_latent = criterion_Latent(xe.float(), xye.float()) + criterion_Latent(ye.float(), yxe.float())
            g_loss_cycle = criterion_Cycle(xe, xeGyeFxe) + criterion_Cycle(ye, yeFxeGye) 
            g_loss = g_loss_gan*1 + g_loss_cycle*0.01 + g_loss_commitment*0.1
            
            # g_loss = g_loss_recon*1000
            optim_gen.zero_grad()
            g_loss.backward()
            if Distributed_Flag==False or local_rank<=0:
                Write_tensorboard('genGe', list(genGe.named_parameters()), epoch_i * len_dataset/2 + loader_i)# 记录梯度信息
                Write_tensorboard('genFe', list(genFe.named_parameters()), epoch_i * len_dataset/2 + loader_i)# 记录梯度信息
            optim_gen.step()    

            # if(loader_i%30==0 and (Distributed_Flag==False or local_rank<=0)):
            #     with torch.no_grad():
            #         # print(disG(xy), disF(yx))
            #         # print(f'g_loss_gan_G {g_loss_gan_G.item():.4e} g_loss_gan_F {g_loss_gan_F.item():.4e}')
            #         # print(f'd_loss_G_real {d_loss_G_real.item():.4e} d_loss_G_fake {d_loss_G_fake.item():.4e} d_loss_F_real {d_loss_F_real.item():.4e} d_loss_F_fake {d_loss_F_fake.item():.4e}')
            #         x_stack = torch.stack((xy,yx,y,xy_,x,yx_), dim = 0)
            #         x_new = einops.rearrange(x_stack, 'x n c h w -> (x h) (n w) c')
            #         x_new = (x_new.clip(-1, 1) + 1) / 2 * 255
            #         x_new = x_new.cpu().numpy().astype(np.uint8)
            #         # print(x_new.shape)
            #         if(x_new.shape[2]==3):
            #             x_new = cv2.cvtColor(x_new, cv2.COLOR_RGB2BGR)
            #         cv2.imwrite(os.path.join(save_dir,'cycle_tmp.jpg'), x_new)
            #     desc=f"g_loss_gan_G {g_loss_gan_G.item():.4e} d_loss_G_real: {d_loss_G_real.item():.4e} d_loss_G_fake: {d_loss_G_fake.item():.4e}"
            #     tqdm.write(desc)


            # g_loss_latent = torch.tensor(0.).to(device)
            # g_loss_cycle = torch.tensor(0.).to(device)
            # g_loss_gan = torch.tensor(0.).to(device)
            # d_loss = torch.tensor(0.).to(device)
            # d_loss_G_real = torch.tensor(0.).to(device)
            # d_loss_G_fake = torch.tensor(0.).to(device)
            # d_loss_F_real = torch.tensor(0.).to(device)
            # d_loss_F_fake = torch.tensor(0.).to(device)
            loss_list += torch.stack([g_loss, d_loss, g_loss_gan, g_loss_cycle, g_loss_commitment, torch.tensor(0.).to(device), torch.tensor(0.).to(device), d_loss_G_real, torch.tensor(0.).to(device), d_loss_F_real, torch.tensor(0.).to(device)])  #把列表转换成tensor
            if Distributed_Flag==False or local_rank<=0:
                step = epoch_i * len_dataset/2 + loader_i
                if(step%180==0):
                    step = int(step/180)
                    writer.add_scalars('g_loss_gan', {'G': g_loss_gan_G, 'F': g_loss_gan_F}, step)
                    writer.add_scalar('g_loss_cycle', g_loss_cycle, step)
                    writer.add_scalar('g_loss_commitment', g_loss_commitment, step)
                    writer.add_scalars('d_loss_wgan', {'G': d_loss_G_real, 'F': d_loss_F_real}, step)
            
            
            
        if Distributed_Flag and local_rank >= 0:
            torch.distributed.barrier()
        loss_list = loss_list/len_dataset    
        toc = time.time()    
        if Distributed_Flag==False or local_rank<=0:
            gan_weights = {'genGe': genGe.module.state_dict(), 'genFe': genFe.module.state_dict(), 'disG': disG.module.state_dict(), 'disF': disF.module.state_dict()}
            torch.save(gan_weights, ckpt_path)
            print(f'epoch {epoch_i} g_loss: {loss_list[0].item():.4e} d_loss: {loss_list[1].item():.4e} time: {(toc - tic):.2f}s')
            print(f'g_loss_gan {loss_list[2].item():.4e} g_loss_cycle: {loss_list[3].item():.4e} g_loss_commitment: {loss_list[4].item():.4e} g_loss_latent: {loss_list[5].item():.4e} g_loss_recon: {loss_list[6].item():.4e}')
            print(f'd_loss_G_real {loss_list[7].item():.4e} d_loss_G_fake: {loss_list[8].item():.4e} d_loss_F_real: {loss_list[9].item():.4e} d_loss_F_fake: {loss_list[10].item():.4e}')
            if(epoch_i%1==0):
                sample(genGe.module, genFe.module, vqvaex.module, vqvaey.module, device=device)


def Write_tensorboard(tag, named_param, step):
    if(step%180==0):
        # 记录梯度信息
        step = int(step/180)
        for name, param in named_param:
            writer.add_histogram(tag + '/grad/'+name, param.grad, step)
            writer.add_histogram(tag + '/param/'+name, param, step)
            # if('model.10.weight' in name):
            #     print(param.grad)

sample_time = 0

def sample(genGe: Gen3, genFe: Gen3, vqvaex:VQVAE, vqvaey:VQVAE, device='cuda'):
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
        xe = vqvaex.encoder(x)
        x_hat = vqvaex.decoder(xe)
        ye = vqvaey.encoder(y)
        y_hat = vqvaey.decoder(ye)
        xeGye = genGe(xe)
        yeFxe = genFe(ye)
        xy = vqvaey.decoder(xeGye)
        yx = vqvaex.decoder(yeFxe)
        xeGyeFxe = genFe(xeGye)
        yeFxeGye = genGe(yeFxe)
        xyx = vqvaex.decoder(xeGyeFxe)
        yxy = vqvaey.decoder(yeFxeGye) 
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

save_dir = './data/face2anime64_noise3/'

if __name__ == '__main__':

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

    if(local_rank ==0) :
        writer = SummaryWriter('./runs')

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
    n_embedding = 128
    dim = 256

    vqvaex = VQVAE(image_shape[0], dim = dim, n_embedding = n_embedding).to(device)
    vqvaey = VQVAE(image_shape[0], dim = dim, n_embedding = n_embedding).to(device)
    genGe = Gen3(dim, 11).to(device)
    genFe = Gen3(dim, 11).to(device)
    # disG = Discriminator(image_shape).to(device)
    # disF = Discriminator(image_shape).to(device)
    # disG = Dis2([dim,16,16]).to(device)
    # disF = Dis2([dim,16,16]).to(device)
    disG = Dis2(image_shape).to(device)
    disF = Dis2(image_shape).to(device)
    # print(disG)
    disG.apply(weights_init_normal)
    disF.apply(weights_init_normal)
    genGe.apply(weights_init_normal)
    genFe.apply(weights_init_normal)
    ckptx_path = 'zqindex/model_vqvae_selfish_faces64_noise.pth'
    ckpty_path = 'zqindex/model_vqvae_anime_faces64_noise.pth'
    vqvaex.load_state_dict(torch.load(ckptx_path))
    vqvaey.load_state_dict(torch.load(ckpty_path))

    # gan_weights = torch.load(ckpt_path)
    # genGe.load_state_dict(gan_weights['vqvaex'])
    # genFe.load_state_dict(gan_weights['genFe'])
    # disG.load_state_dict(gan_weights['disG'])
    # disF.load_state_dict(gan_weights['disF'])
    # sample(vqvaex, vqvaey, device=device)

    if Distributed_Flag:
        total_rank = dist.get_world_size()
        vqvaex = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vqvaex).to(device)
        vqvaex = DDP(vqvaex, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        vqvaey = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vqvaey).to(device)
        vqvaey = DDP(vqvaey, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        disG = torch.nn.SyncBatchNorm.convert_sync_batchnorm(disG).to(device)
        disG = DDP(disG, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        disF = torch.nn.SyncBatchNorm.convert_sync_batchnorm(disF).to(device)
        disF = DDP(disF, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        genGe = torch.nn.SyncBatchNorm.convert_sync_batchnorm(genGe).to(device)
        genGe = DDP(genGe, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        genFe = torch.nn.SyncBatchNorm.convert_sync_batchnorm(genFe).to(device)
        genFe = DDP(genFe, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    

    train(genGe, genFe, vqvaex, vqvaey, disG, disF ,local_rank, ckpt_path, device=device)

