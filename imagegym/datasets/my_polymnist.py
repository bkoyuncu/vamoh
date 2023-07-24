from typing import Callable, Optional

import numpy as np
import torch
from torchvision.datasets import MNIST, VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
import os

class MyPolyMNIST(VisionDataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            missing_perc=0.0,
            noise_value=0.0,
            use_one_hot=False,
            modality= "m0"
    )-> None:
            super().__init__(root, transform= transform, target_transform= target_transform)

            self.train = train  # training set or test set
            self.modality = modality
            self.data, self.targets = self._load_data()
            self.noise = 0.0
            self.mask = None
            self.use_one_hot = use_one_hot
            self.missing_percent = missing_perc
            self.missing_data = True if missing_perc > 0 else False
    

    def _load_data(self):
    # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        
        if self.train:
            folder="train"
        else:
            folder="test"

        data_path = os.path.join(self.root, folder, self.modality + str(".pt"))
        target_path = os.path.join(self.root, folder, "labels.pt")
        data= torch.load(data_path) #torch.Size([60000, 3, 28, 28])
        targets= torch.load(target_path) #torch.Size([60000])
        return data, targets
        # return torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, concepts) where target is index of the target class and concepts is a one-hot vector
        """
        img, target = self.data[index], int(self.targets[index])
        img = img.numpy() #[3,28,28] #between [0,1]

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)

        if self.target_transform is not None:
            target = self.target_transform(target)


        if not isinstance(self.noise, float):
            target = self.noise[index]

        if self.use_one_hot:
            target_out = torch.zeros(10)
            target_out[target] = 1.0
        else:
            target_out = target
       
        return img, target_out


    def __len__(self) -> int:
        return len(self.data)
