from typing import Callable, Optional

import numpy as np
import torch
from torchvision.datasets import MNIST, VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader


class ERA5Dataset(Dataset):
    """ERA5 temperature dataset.
    Args:
        path_to_data (string): Path to directory where data is stored.
        transform (torchvision.Transform): Optional transform to apply to data.
        normalize (bool): Whether to normalize data to lie in [0, 1]. Defaults
            to True.
    """
    def __init__(self, root, transform=None, split="train", normalize=True, missing_perc=0.0):
        self.root = root
        self.transform = transform
        self.normalize = True
        self.split = split
        self.data = self._load_data()
        # t = torch.rand(4, 2, 3, 3)
        idx = torch.randperm(self.data.shape[0])
        self.data = self.data[idx].view(self.data.size())
        # print(torch.mean(self.data[:,2]))
        self.missing_percent = missing_perc
        self.missing_data = missing_perc > 0

    def _load_data(self):
    # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        
        if self.split == 'train':
            file_name="tensor_train_rnd"
            
        elif self.split == 'val':
            file_name="tensor_val_rnd"
            
        elif self.split == 'test':
            file_name="tensor_test_rnd"

        data_path = os.path.join(self.root, "era5", file_name + str(".pt"))
        # target_path = os.path.join(self.root, folder, "labels.pt")
        data= torch.load(data_path) #torch.Size([60000, 3, 28, 28])
        # targets= torch.load(target_path) #torch.Size([60000])
        return data
        # return torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        data_tensor = self.data[index]
        data_tensor = data_tensor.numpy() # [3,46,90]
        if self.transform:
            data_tensor = self.transform(data_tensor)
    
        return data_tensor, 0  # Label to ensure consistency with image datasets
    
    def __len__(self) -> int:
        return len(self.data)
