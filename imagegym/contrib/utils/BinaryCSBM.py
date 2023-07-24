from typing import Optional, Callable, List

import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from imagegym.contrib.utils.random import binary_contextual_stochstic_blockmodel_graph
from torch_geometric.data import Data


class BCSBMDataset(InMemoryDataset):

    r""" Binary Contextual Stochastic  Blockmodel Graph  (BCSBG) Dataset
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 n: int = 100,
                 eps: float = 0.5,
                 p: float = 0.7,
                 q: float = 0.4,
                 mu: list = [1],
                 std_dev: float = 1.0,
                 directed: bool = False,
                 seed: int = 0,
                 num_val: int = 500,
                 num_test: int = 1000,
                 split: str = "full"):


        assert n > 0
        assert q > 0 and q < 1
        assert p > 0 and p < 1
        assert len(mu) > 0
        assert std_dev > 0
        assert 2*(num_val + num_test)  < n

        self.name = 'BCSGB'
        self.n = n
        self.eps = eps
        self.p = p
        self.q = q
        self.mu = torch.FloatTensor(mu)
        self.std_dev = std_dev
        self.directed = directed
        self.seed = seed

        super().__init__(root, transform, pre_transform, pre_filter)
        o = binary_contextual_stochstic_blockmodel_graph(n=self.n,
                                                         p=self.p,
                                                         q=self.q,
                                                         mu=self.mu,
                                                         std_dev=self.std_dev,
                                                         eps=self.eps,
                                                         directed=self.directed)

        edge_index, label, node_features, edge_label = o

        data = Data(x=node_features, edge_index=edge_index, edge_attr=None, y=label,
                    edge_label=edge_label)

        self.data = data if self.pre_transform is None else self.pre_transform(data)
        self.data.train_mask = torch.full([self.n], True)
        self.data.val_mask = torch.full([self.n], False)
        self.data.test_mask = torch.full([self.n], False)

        self.split = split
        assert self.split in ['full', 'random']

        if split == 'full':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            data.train_mask.fill_(False)
            num_train = data.x.shape[0] - num_val - num_test
            num_train_per_class = num_train // self.num_classes

            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True
            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')


    @property
    def num_node_labels(self) -> int:
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self) -> List[str]:
        return ['nothing']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
