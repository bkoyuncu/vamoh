import torch
import torchvision
import os
from imagegym.config import cfg
from imagegym.contrib.utils.random import get_permutation
from torch.utils.data import DataLoader


def compute_split_idx(original_len, split_sizes, random=True):
    all_idx = torch.arange(original_len)
    if random:
        perm = get_permutation(original_len=original_len)
        all_idx = all_idx[perm]

    start_idx, end_idx = 0, None
    all_idx_splits = []

    num_splits = len(split_sizes)
    for i, size in enumerate(split_sizes):
        assert isinstance(size, float)
        assert 0 < size
        assert 1 > size
        new_len = int(size * original_len)
        end_idx = new_len + start_idx
        if i == (num_splits - 1):
            all_idx_splits.append(all_idx[start_idx:])
        else:
            all_idx_splits.append(all_idx[start_idx:end_idx])
        start_idx = end_idx

    return all_idx_splits


def transform_after_split(datasets):
    '''
    Dataset transformation after train/val/test split
    :param dataset: A list of DeepSNAP dataset objects
    :return: A list of transformed DeepSNAP dataset objects
    '''

    return datasets


def load_torch(name, dataset_dir):
    '''
    load pyg format dataset
    :param name: dataset name
    :param dataset_dir: data directory
    :return: a list of networkx/deepsnap graphs
    '''
    dataset_dir = os.path.join(dataset_dir, name)  # './datasets/MNIST'
    print(dataset_dir)
    datasets = []
    if name in ['PolyMNIST']:
        from imagegym.datasets.my_polymnist import MyPolyMNIST
        from torchvision import transforms
        
        dataset_train = MyPolyMNIST(root=dataset_dir, train=True,
                                    download=False,
                                    missing_perc=cfg.dataset.missing_perc,
                                    use_one_hot=cfg.dataset.use_one_hot,
                                    modality=cfg.dataset.modality)

        setattr(dataset_train, 'num_classes', 10)
        dataset_test = MyPolyMNIST(root=dataset_dir, train=False,
                                   download=False,
                                   missing_perc=cfg.dataset.missing_perc,
                                   use_one_hot=cfg.dataset.use_one_hot,
                                   modality=cfg.dataset.modality)

        cfg.dataset.dims = [3, 28, 28]
        cfg.dataset.label_dim = 10
        cfg.dataset.coordinate_dim = 2

        datasets.append(dataset_train)
        datasets.append(dataset_test)

    elif name in ['shapes3d', 'shapes3d_10','shapes3d_50']:
        from imagegym.datasets.shapes3d import Shapes3D
        from torchvision import transforms
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
        if name != 'shapes3d':
            percentage = int(name.replace('shapes3d_', ''))
        else:
            percentage = 100
        dataset_train = Shapes3D(root=dataset_dir,
                                 split='train',
                                 transform=transform,
                                 seed=0,
                                 percentage=percentage,
                                 download=False,
                                 load_from_numpy=True,
                                 missing_perc=cfg.dataset.missing_perc)
        setattr(dataset_train, 'num_classes', 6)
        dataset_valid = Shapes3D(root=dataset_dir,
                                 split='val',
                                 transform=transform,
                                 seed=0,
                                 percentage=percentage,
                                 download=False,
                                 load_from_numpy=True,
                                 missing_perc=cfg.dataset.missing_perc)
        dataset_test = Shapes3D(root=dataset_dir,
                                split='test',
                                transform=transform,
                                seed=0,
                                percentage=percentage,
                                download=False,
                                load_from_numpy=True,
                                missing_perc=cfg.dataset.missing_perc)

        cfg.dataset.dims = [3, 64, 64]
        cfg.dataset.label_dim = 6
        cfg.dataset.coordinate_dim = 2

        datasets.append(dataset_train)
        datasets.append(dataset_valid)
        datasets.append(dataset_test)
    
    elif name in ['celebahq256']:
        from imagegym.datasets.my_celebahq import MyCelebAHQ
        from torchvision import transforms
        size = cfg.dataset.size
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            transforms.Resize((size, size))
            ])

        dataset_train = MyCelebAHQ(root=dataset_dir, # should be 'datasets/celebahq256/'
                                 split='train',
                                 transform=transform,
                                 download=False,
                                 missing_perc=cfg.dataset.missing_perc,
                                 use_one_hot=cfg.dataset.use_one_hot)

        setattr(dataset_train, 'num_classes', 40)
        dataset_valid = MyCelebAHQ(root=dataset_dir, # should be 'datasets/celebahq256/'
                                 split='valid',
                                 transform=transform,
                                 download=False,
                                 missing_perc=cfg.dataset.missing_perc,
                                 use_one_hot=cfg.dataset.use_one_hot)
        dataset_test = MyCelebAHQ(root=dataset_dir, # should be 'datasets/celebahq256/'
                                split='test',
                                transform=transform,
                                download=False,
                                missing_perc=cfg.dataset.missing_perc,
                                use_one_hot=cfg.dataset.use_one_hot)

        cfg.dataset.dims = [3, size, size]
        cfg.dataset.label_dim = 40
        cfg.dataset.coordinate_dim = 2

        datasets.append(dataset_train)
        datasets.append(dataset_valid)
        datasets.append(dataset_test)

    
    elif name in ["voxels"]:
        from imagegym.datasets.my_pointcloud import VoxelDataset

        dataset_train = VoxelDataset(root=dataset_dir,
                                    split='train')
        dataset_valid = VoxelDataset(root=dataset_dir,
                                    split='val')
        dataset_test = VoxelDataset(root=dataset_dir,
                                    split='test')

        cfg.dataset.dims = [1, 32, 32, 32]
        cfg.dataset.label_dim = 0
        cfg.dataset.coordinate_dim = 3

        datasets.append(dataset_train)
        datasets.append(dataset_valid)
        datasets.append(dataset_test)
    
    elif name in ['era5']:
        from imagegym.datasets.my_era5 import ERA5Dataset

        dataset_train = ERA5Dataset(root=cfg.dataset.dir,
                                    split='train',
                                    transform=None,
                                    missing_perc=cfg.dataset.missing_perc)

        dataset_valid = ERA5Dataset(root=cfg.dataset.dir,
                                    split='val',
                                    transform=None,
                                    missing_perc=cfg.dataset.missing_perc)

        dataset_test = ERA5Dataset(root=cfg.dataset.dir,
                                    split='test',
                                    transform=None,
                                    missing_perc=cfg.dataset.missing_perc)

        #between [0,1]
        #real shape 3,46,90 but first two are coordinates
        cfg.dataset.dims = [1, 46, 90]
        cfg.dataset.label_dim = 0
        cfg.dataset.coordinate_dim = 3 #spherical

        datasets.append(dataset_train)
        datasets.append(dataset_valid)
        datasets.append(dataset_test)

    else:
        raise ValueError('{} not support'.format(name))

    assert cfg.dataset.dims is not None
    assert cfg.dataset.label_dim is not None
    return datasets


def load_dataset():
    '''
    load raw datasets.
    :return: a list of networkx/deepsnap graphs, plus additional info if needed
    '''
    format = cfg.dataset.format  # torch
    name = cfg.dataset.name
    # dataset_dir = '{}/{}'.format(cfg.dataset.dir, name)
    dataset_dir = cfg.dataset.dir  # './datasets'
    # Load from Pytorch Geometric dataset
    if format == 'torch':
        datasets = load_torch(name, dataset_dir)
    else:
        raise ValueError('Unknown data format: {}'.format(cfg.dataset.format))

    return datasets


def filter_samples(datasets):
    return datasets


def create_dataset():
    ## Load dataset
    datasets = load_dataset()
    datasets = transform_after_split(datasets)  # empty
    return datasets


def create_loader(datasets):
    
    if cfg.dataset.use_train_as_valid:
        train_shuffle = False
    else:
        train_shuffle = True

    # if cfg.dataset.check_data or cfg.train.mode == "sample":
    #     train_shuffle = False

    loader_train = DataLoader(datasets[0],
                              batch_size=cfg.train.batch_size, shuffle=train_shuffle,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    loaders = [loader_train]
    for i in range(1, len(datasets)):
        if cfg.dataset.use_train_as_valid:
            loaders.append(DataLoader(datasets[0],
                                      batch_size=cfg.train.batch_size, shuffle=False,
                                      num_workers=cfg.num_workers, pin_memory=cfg.pin_memory))
        else:
            loaders.append(DataLoader(datasets[i],
                                      batch_size=cfg.train.batch_size,
                                      shuffle=False,
                                      num_workers=cfg.num_workers,
                                      pin_memory=cfg.pin_memory))

    return loaders
