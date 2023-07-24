import os
import os.path
import random
import time
from typing import Any, Callable, Dict, Optional, Tuple

import h5py
import numpy as np
import requests
import torch
from torchvision.datasets.vision import VisionDataset

from imagegym.utils.io import print_info, print_warning


class Shapes3D(VisionDataset):
    _H5_FILENAME = '3dshapes.h5'
    _DATA_FILENAME = 'data.npy'
    _DATA_FILENAME_SPLIT = lambda x: f'data_{x}.npy'
    _TARGETS_FILENAME = 'targets.npy'
    _TARGETS_FILENAME_SPLIT = lambda x: f'targets_{x}.npy'
    _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                         'orientation']
    _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                              'scale': 8, 'shape': 4, 'orientation': 15}

    URL = 'https://storage.googleapis.com/3d-shapes/3dshapes.h5'

    def __init__(
            self,
            root: str,
            split: str = 'train',
            seed: int = 0,
            percentage: int = 100,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            load_from_numpy: bool = False,
            missing_perc:float =0.0,
    ) -> None:
        super(Shapes3D, self).__init__(root, transform=transform,
                                       target_transform=target_transform)
        assert split in ['train', 'val', 'test']
        self.split = split  # training set or test set
        self.seed = seed
        self.percentage = percentage
        self.load_from_numpy = load_from_numpy
        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self.data, self.targets = self._load_data()
        self.missing_percent = missing_perc
        self.missing_data = True if missing_perc > 0 else False

    def _download(self):

        filename_h5 = os.path.join(self.root, Shapes3D._H5_FILENAME)
        print_info(f'Downloading the data from {Shapes3D.URL} and saving it in {filename_h5}')

        if os.path.exists(filename_h5):
            print_warning('Aborting download, the file is already in the folder!')
            print_info('Delete the file if you want to download it again')
            return
        try:
            response = requests.get(Shapes3D.URL)
            with open(filename_h5, "wb") as f:
                f.write(response.content)
        except:
            print_warning('Downloading data failed!')
            print_info(f'Download the data manually from {Shapes3D.URL} and put it in {self.root}')
        return

    def _filename_data(self, split=None):
        if split is None: split = self.split
        if self.percentage == 100:
            basename = Shapes3D._DATA_FILENAME_SPLIT(f"{split}_{self.seed}")
        else:
            basename = Shapes3D._DATA_FILENAME_SPLIT(f"{split}_{self.seed}_{self.percentage}")
        filename_data = os.path.join(self.root, basename)
        return filename_data

    def _filename_targets(self, split=None):
        if split is None: split = self.split
        if self.percentage == 100:
            basename = Shapes3D._TARGETS_FILENAME_SPLIT(f"{split}_{self.seed}")
        else:
            basename = Shapes3D._TARGETS_FILENAME_SPLIT(f"{split}_{self.seed}_{self.percentage}")
        filename_targets = os.path.join(self.root, basename)
        return filename_targets

    def _load_data(self):
        filename_data = self._filename_data()
        filename_targets = self._filename_targets()

        files_exist = all([os.path.exists(f) for f in [filename_data, filename_targets]])
        if files_exist and self.load_from_numpy:
            print_info(f'Loading data: {filename_data}')
            data = np.load(filename_data)
            print_info(f'Loading targets: {filename_targets}')
            targets = np.load(filename_targets)
            return data, targets
        else:
            filename_h5 = os.path.join(self.root, Shapes3D._H5_FILENAME)
            dataset = h5py.File(filename_h5, 'r')
            data_h5 = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
            targets_h5 = dataset['labels']  # array shape [480000,6], float64

            if self.load_from_numpy:
                time_1 = time.time()
                print_info(f'Creating data: {filename_data}')
                data = data_h5[:]
                time_2 = time.time()
                print_info(f'Done creating data: {time_2 - time_1}')
                print_info(f'Creating targets: {filename_data}')
                targets = targets_h5[:]
                time_3 = time.time()
                print_info(f'Done creating targets: {time_3 - time_2}')
                print_info(f'Total time: {time_3 - time_1}')
                del dataset

                idx_ = list(range(data.shape[0]))
                random.seed(self.seed)
                random.shuffle(idx_)
                idx_ = torch.tensor(idx_)
                num_images = data.shape[0]
                split_sections = [int(num_images * p) for p in [0.8, 0.1, 0.1]]
                split_sections[-1] = num_images - sum(split_sections[:-1])
                idx_list_all = torch.split(idx_, split_size_or_sections=split_sections)

                splits = ['train', 'val', 'test']
                idx_list = []
                for i, split in enumerate(splits):
                    if self.percentage < 100:
                        num_ = int(len(idx_list_all[i]) * self.percentage / 100)
                        idx_list_i = idx_list_all[i][:num_]
                    else:
                        idx_list_i = idx_list_all[i]
                    idx_list.append(idx_list_i.tolist())

                for i, split in enumerate(splits):
                    filename_data = self._filename_data(split=split)
                    filename_targets = self._filename_targets(split=split)
                    idx_list_i = idx_list[i]

                    np.save(filename_data, data[idx_list_i])
                    print_info(f'Saving data: {filename_data}')
                    np.save(filename_targets, targets[idx_list_i])
                    print_info(f'Saving targets: {filename_targets}')

                idx = idx_list[splits.index(self.split)]

                return data[idx], targets[idx]
            else:
                return data_h5, targets_h5

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(Shapes3D._FACTORS_IN_ORDE)}

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.root, Shapes3D._H5_FILENAME))

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


def compare_performance_h5_npy(root):
    for load_from_numpy in [True, False]:
        time_0 = time.time()
        dataset = Shapes3D(root=root, split='train', load_from_numpy=load_from_numpy)
        time_1 = time.time()
        print_info(f"[load_from_numpy={load_from_numpy}] Shapes3D {time_1 - time_0}")

        for i in range(100):
            batch = dataset.__getitem__(i)

        time_2 = time.time()
        print_info(f"[load_from_numpy={load_from_numpy}] __getitem__ {time_2 - time_1}")
        print('')
