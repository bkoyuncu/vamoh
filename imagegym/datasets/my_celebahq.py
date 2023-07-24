from typing import Callable, Optional

import numpy as np
import torch
from torchvision.datasets import CelebA


class MyCelebAHQ(CelebA):


    def __init__(
            self,
            root: str,
            split: str = 'train',
            target_type: str = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            missing_perc=0.0,
            noise_value=0.0,
            use_one_hot=False,
    ):
        super().__init__(root=root,
                         split=split ,
                         target_type=target_type,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.noise = 0.0
        self.mask = None
        self.use_one_hot = use_one_hot
        self.missing_percent = missing_perc
        self.missing_data = True if missing_perc > 0 else False

    def get_cond_prior(self):
        return torch.distributions.Bernoulli(probs=[1/40,]*40)

    @property
    def dim_cond(self):
        return 40

    @property
    def distr_name_cond(self):
        return 'b'

    # This shouldnt be done but we need it for now
    def _check_integrity(self) -> bool:
        """for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))"""
        return True

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, concepts) where target is index of the target class and concepts is a one-hot vector
        """
        img, target = super().__getitem__(index) #in [0,1]

        return img, target
