from typing import Callable, Optional

import numpy as np
import torch
from torchvision.datasets import MNIST


class MyMNIST(MNIST):


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
    ):
        super().__init__(root, train, transform, target_transform, download)

        self.noise = 0.0
        self.mask = None
        self.use_one_hot = use_one_hot
        self.missing_percent = missing_perc
        self.missing_data = True if missing_perc > 0 else False
    def get_cond_prior(self):
        return torch.distributions.Categorical(probs=[1/10,]*10)

    @property
    def dim_cond(self):
        return 10

    @property
    def distr_name_cond(self):
        return 'cat'

    def set_noise(self, noise_value):
        if noise_value > 0.0:
            mask_noise = torch.distributions.Bernoulli(1 - noise_value).sample(self.targets.size()).long() == 0
            label_noise = torch.distributions.Categorical(probs=[1/10,]*10).sample([mask_noise.sum()])

            self.noise = self.targets.clone()
            self.noise[mask_noise] = label_noise
        else:
            self.noise = 0.0

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, concepts) where target is index of the target class and concepts is a one-hot vector
        """
        img, target = self.data[index], int(self.targets[index])
        img = img.numpy() #[28,28]

        if self.transform is not None:
            img = self.transform(img) #thresholding works

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
