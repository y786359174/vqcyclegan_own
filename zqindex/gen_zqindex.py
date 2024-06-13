import os
import torch
import sys
sys.path.append('../')
from vqvae.vqvae import VQVAE
from CycleGAN_own.dataset import get_dataloader, get_img_shape

Imgx_dir_name = '../face2face/dataset/train/A'
dataloaderx = get_dataloader(1,Imgx_dir_name)
# 设置保存路径
save_dir = 'zqindex/selfish_faces64_xi'
os.makedirs(save_dir, exist_ok=True)
image_shape = get_img_shape()
dim = 256
n_embedding = 128
device = 'cuda'
vqvaex = VQVAE(image_shape[0], dim = dim, n_embedding = n_embedding).to(device)
ckptx_path = 'zqindex/model_vqvae_selfish_faces64.pth'
vqvaex.load_state_dict(torch.load(ckptx_path))
# 训练循环
batch_idx = 0
for x, _ in dataloaderx:
    batch_idx = batch_idx + 1
    x = x.to(device)
    xi = vqvaex.encode(x)
    
    # 保存 xi
    torch.save(xi, os.path.join(save_dir, f'xi_{batch_idx}.pt'))