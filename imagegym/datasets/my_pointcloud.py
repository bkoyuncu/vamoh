from typing import Callable, Optional

import numpy as np
import torch
from torchvision.datasets import CelebA
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import os

class My_PointCloudDataset(Dataset):
    """Three dimensional point cloud datasets. Each datapoint is a tensor of shape
    (num_points, 4), where the first 3 columns correspond to the x, y, z coordinates 
    of the point and the fourth column is a label (0 or 1) corresponding to whether
    the point is inside or outside the object.
    Args:
        data_dir (torch.utils.Dataset): Path to directory where point cloud data 
            is stored.
    """
    def __init__(self, data_dir, split = "train", missing_perc=0.0):
        data_split = "tensor_" + split + ".pt"
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, data_split)
        self.data = torch.load(self.path)
        self.missing_percent = missing_perc
        self.missing_data = missing_perc > 0
        

    def __getitem__(self, index):
        #TODO add missingness
        # Shape (4, num_points) old: Shape (num_points, 4)
        point_cloud = self.data[index]
        # Change coordinates [-.5, .5] -> [-1., 1.]
        # point_cloud[:3]= 2. * point_cloud[:3]
        new_point_cloud = torch.cat((2. * point_cloud[:,:3],point_cloud[:,[-1]]),dim=1) #it is mutable otherwise it changes the data

        if self.missing_data:
            if self.missing_percent == 1:
                missing_rate = np.random.rand(1) * 0.9  
            elif self.missing_percent > 0:
                missing_rate = self.missing_percent
                missing_rate = np.random.uniform(low=0,high=missing_rate)
            elif self.missing_percent ==0:
                missing_rate = -1
            observed_mask_0 = (np.random.rand(new_point_cloud.shape[0]) > missing_rate)
            observed_mask = np.repeat(observed_mask_0[:,np.newaxis], new_point_cloud.shape[1], axis=1)
            return new_point_cloud, 0, observed_mask
        else:
            return new_point_cloud, 0  # Return unused label to match image datasets

    def __len__(self):
        return len(self.data)



class VoxelDataset(Dataset):
    """Three dimensional voxel datasets.

    Args:
        data_dir (torch.utils.Dataset): Path to directory where voxel data is
            stored.
        size (int): Size of voxel cube side.
        threshold (float): If interpolating, threshold to use to determine voxel
            occupancy. Works best with low values.
    """

    def __init__(self,
                 root,
                 split='train',
                 size=32,
                 threshold=0.05,
                 include_idx: bool = False):
        self.root = root
        self.voxel_paths = glob.glob(root + "/*.pt")
        self.split = split
        self.include_idx = include_idx

        num_samples = len(self.voxel_paths)

        self.split_sizes = {
            'train': 0.8,
            'val': 0.1,
            'test': 0.1
        }

        num_samples_tr = int(0.8 * num_samples)
        num_samples_val = int(0.1 * num_samples)
        num_samples_tst = num_samples - (num_samples_tr + num_samples_val)
        self.split_sizes = {
            'train': num_samples_tr,
            'val': num_samples_val,
            'test': num_samples_tst
        }

        if split == 'train':
            self.start_index = 0
        else:
            if split == 'val':
                self.start_index = num_samples_tr
            elif split == 'test':
                self.start_index = num_samples_tr + num_samples_val

        self.voxel_paths.sort()  # Ensure consistent ordering of voxels
        self.size = size
        self.threshold = threshold
        self.missing_data = True #It is true because we are always sampling mask


    def __getitem__(self, index):
        # Shape (depth, height, width)
        voxels = torch.load(self.voxel_paths[self.start_index + index])
        # Unsqueeze to get shape (1, depth, height, width)
        voxels = voxels.unsqueeze(0).float()  #torch.Size([1, 32, 32, 32])
        # Optionally resize
        if self.size != 32:
            # Need to add batch dimension for interpolate function
            voxels = torch.nn.functional.interpolate(voxels.unsqueeze(0).float(), self.size,
                                                     mode='trilinear')[0]
            # Convert back to byte datatype
            voxels = voxels > self.threshold
    
        return voxels, 0  # Return unused label to match image datasets

    def __len__(self):
        return self.split_sizes[self.split]

    def random_indices(self, num_indices, max_idx):
        """Generates a set of num_indices random indices (without replacement)
        between 0 and max_idx - 1.

        Args:
            num_indices (int): Number of indices to include.
            max_idx (int): Maximum index.
        """
        # It is wasteful to compute the entire permutation, but it looks like
        # PyTorch does not have other functions to do this
        permutation = torch.randperm(max_idx)
        # Select first num_indices indices (this will be random since permutation is
        # random)
        return permutation[:num_indices]
