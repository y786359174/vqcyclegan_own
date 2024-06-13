import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class XiDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.xi_files = [f for f in os.listdir(root_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.xi_files)

    def __getitem__(self, idx):
        xi_path = os.path.join(self.root_dir, self.xi_files[idx])
        xi = torch.load(xi_path)
        return xi

def xiget_dataloader(batch_size: int, data_dir, num_workers = 0, distributed = False):
    
    dataset = XiDataset(root_dir=data_dir)

    if distributed:
        sampler = DistributedSampler(dataset)
    
    if distributed:
        return DataLoader(
            dataset, 
            batch_size=batch_size,
            num_workers = num_workers,
            shuffle=(sampler is None),
            sampler = sampler, 
            pin_memory=False,   # Tensor long 没办法存进去
            drop_last=True,
             )
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers, drop_last=True)

def xiget_img_shape():
    # return (3, 256, 256)    # 虽然这么写很蠢。但是好像还真挺好用
    return (1, 16, 16)    # 虽然这么写很蠢。但是好像还真挺好用