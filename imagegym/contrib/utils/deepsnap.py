import re
import types
import random
import copy
import math
import pdb
import numpy as np
import torch
from torch_geometric.utils import to_undirected
from typing import (
    Dict,
    List,
    Union
)
import warnings
import deepsnap

from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset

from imagegym.config import cfg

class MyGraphDataset2(GraphDataset):
    def __init__(self,*args, **kwargs):
        super(MyGraphDataset2, self).__init__(*args, **kwargs)
    @property
    def num_graph_labels(self) -> int:
        r"""
        Returns the number of graph labels.

        Returns:
            int: The number of graph labels for the graphs
            in the dataset.
        """
        if self._num_graph_labels is None:
            if self.graphs is None:
                self._num_graph_labels = self.generator.num_graph_labels
            else:
                if cfg.dataset.label_dim is not None:
                    self._num_graph_labels = cfg.dataset.label_dim
                else:
                    unique_graph_labels = torch.LongTensor([])
                    for graph in self.graphs:
                        unique_graph_labels = torch.cat([
                            unique_graph_labels,
                            graph.get_num_labels("graph_label")
                        ])
                    self._num_graph_labels = torch.unique(
                        unique_graph_labels
                    ).shape[0]
        return self._num_graph_labels
class MyGraphDataset(GraphDataset):
    def __init__(self,*args, **kwargs):
        super(MyGraphDataset, self).__init__(*args, **kwargs)

    @staticmethod
    def pyg_to_graphs(
            dataset,
            verbose: bool = False,
            fixed_split: bool = False,
            tensor_backend: bool = False,
            netlib=None
    ) -> List[Graph]:
        r"""
        Transform a :class:`torch_geometric.data.Dataset` object to a
        list of :class:`deepsnap.grpah.Graph` objects.

        Args:
            dataset (:class:`torch_geometric.data.Dataset`): A
                :class:`torch_geometric.data.Dataset` object that will be
                transformed to a list of :class:`deepsnap.grpah.Graph`
                objects.
            verbose (bool): Whether to print information such as warnings.
            fixed_split (bool): Whether to load the fixed data split from
                the original PyTorch Geometric dataset.
            tensor_backend (bool): `True` will use pure tensors for graphs.
            netlib (types.ModuleType, optional): The graph backend module.
                Currently DeepSNAP supports the NetworkX and SnapX (for
                SnapX only the undirected homogeneous graph) as the graph
                backend. Default graph backend is the NetworkX.

        Returns:
            list: A list of :class:`deepsnap.graph.Graph` objects.
        """

        if fixed_split:
            graphs = [
                MyGraph.pyg_to_graph(
                    data, verbose=verbose, fixed_split=True,
                    tensor_backend=tensor_backend, netlib=netlib
                )
                for data in dataset
            ]
            graphs_split = [[graph] for graph in graphs[0]]
            return graphs_split
        else:
            return [
                MyGraph.pyg_to_graph(
                    data, verbose=verbose,
                    tensor_backend=tensor_backend,
                    netlib=netlib
                )
                for data in dataset
            ]
class MyGraph(Graph):
    def __init__(self,*args, **kwargs):
        super(MyGraph, self).__init__(*args, **kwargs)

    @staticmethod
    def pyg_to_graph(
            data,
            verbose: bool = False,
            fixed_split: bool = False,
            tensor_backend: bool = False,
            netlib=None
    ):
        r"""
        Transform a :class:`torch_geometric.data.Data` object to a
        :class:`Graph` object.

        Args:
            data (:class:`torch_geometric.data.Data`): A
                :class:`torch_geometric.data.Data` object that will be
                transformed to a :class:`deepsnap.grpah.Graph`
                object.
            verbose (bool): Whether to print information such as warnings.
            fixed_split (bool): Whether to load the fixed data split from
                the original PyTorch Geometric data.
            tensor_backend (bool): `True` will use pure tensors for graphs.
            netlib (types.ModuleType, optional): The graph backend module.
                Currently DeepSNAP supports the NetworkX and SnapX (for
                SnapX only the undirected homogeneous graph) as the graph
                backend. Default graph backend is the NetworkX.

        Returns:
            :class:`Graph`: A new DeepSNAP :class:`Graph` object.
        """
        # all fields in PyG Data object
        kwargs = {}
        kwargs["node_feature"] = data.x if "x" in data.keys else None
        kwargs["edge_feature"] = (
            data.edge_attr if "edge_attr" in data.keys else None
        )
        kwargs["node_label"], kwargs["edge_label"] = None, None
        kwargs["graph_feature"], kwargs["graph_label"] = None, None
        if kwargs["node_feature"] is not None and data.y.size(0) == kwargs[
            "node_feature"
        ].size(0):
            kwargs["node_label"] = data.y
        elif kwargs["edge_feature"] is not None and data.y.size(0) == kwargs[
            "edge_feature"
        ].size(0):
            kwargs["edge_label"] = data.y
        else:
            kwargs["graph_label"] = data.y

        if not tensor_backend:
            if netlib is not None:
                deepsnap._netlib = netlib
            G = deepsnap._netlib.DiGraph()
            # if data.is_directed():
            #     G = deepsnap._netlib.DiGraph()
            # else:
            #     G = deepsnap._netlib.Graph()
            G.add_nodes_from(range(data.num_nodes))
            G.add_edges_from(data.edge_index.T.tolist())
        else:
            attributes = {}
            if not data.is_directed():
                row, col = data.edge_index
                mask = row < col
                row, col = row[mask], col[mask]
                edge_index = torch.stack([row, col], dim=0)
                edge_index = torch.cat(
                    [edge_index, torch.flip(edge_index, [0])],
                    dim=1
                )
            else:
                edge_index = data.edge_index
            attributes["edge_index"] = edge_index

        # include other arguments that are in the kwargs of pyg data object
        keys_processed = ["x", "y", "edge_index", "edge_attr"]
        for key in data.keys:
            if key not in keys_processed:
                kwargs[key] = data[key]

        # we assume that edge-related and node-related features are defined
        # the same as in Graph._is_edge_attribute and Graph._is_node_attribute
        for key, value in kwargs.items():
            if value is None:
                continue
            if Graph._is_node_attribute(key):
                if not tensor_backend:
                    Graph.add_node_attr(G, key, value)
                else:
                    attributes[key] = value
            elif Graph._is_edge_attribute(key):
                # TODO: make sure the indices of edge attributes are same with edge_index
                if not tensor_backend:
                    # the order of edge attributes is consistent
                    # with edge index
                    Graph.add_edge_attr(G, key, value)
                else:
                    attributes[key] = value
            elif Graph._is_graph_attribute(key):
                if not tensor_backend:
                    Graph.add_graph_attr(G, key, value)
                else:
                    attributes[key] = value
            else:
                if verbose:
                    print(f"Index fields: {key} ignored.")

        if fixed_split:
            masks = ["train_mask", "val_mask", "test_mask"]
            if not tensor_backend:
                graph = MyGraph(G, netlib=netlib)
            else:
                graph = MyGraph(**attributes)
            if graph.edge_label is not None:
                graph.negative_label_val = torch.max(graph.edge_label) + 1

            graphs = []
            for mask in masks:
                if mask in kwargs:
                    graph_new = copy.copy(graph)
                    graph_new.node_label_index = (
                        torch.nonzero(data[mask]).squeeze()
                    )
                    graph_new.node_label = (
                        graph_new.node_label[graph_new.node_label_index]
                    )
                    graphs.append(graph_new)

            return graphs
        else:
            if not tensor_backend:
                # return Graph(G, netlib=netlib)
                if data.is_directed():
                    return MyGraph(G, netlib=netlib)
                else:
                    return MyGraph(G.to_undirected(), netlib=netlib)
            else:
                graph = Graph(**attributes)
            if graph.edge_label is not None:
                graph.negative_label_val = torch.max(graph.edge_label) + 1
            return graph

    def _split_edge(self, split_ratio: float, shuffle: bool = True):
        r"""
        Split the graph into len(split_ratio) graphs for node prediction.
        Internally this splits node indices, and the model will only compute
        loss for the embedding of nodes in each split graph.
        In edge classification, the whole graph is observed in train/val/test.
        Only split over edge_label_index.
        """
        if self.num_edges < len(split_ratio):
            raise ValueError(
                "In _split_node num of edges are smaller than"
                "number of splitted parts."
            )

        split_graphs = []
        if shuffle:
            shuffled_edge_indices = torch.randperm(self.num_edges)
        else:
            shuffled_edge_indices = torch.arange(self.num_edges)
        split_offset = 0

        # used to indicate whether default splitting results in
        # empty splitted graphs
        split_empty_flag = False
        edges_split_list = []

        for i, split_ratio_i in enumerate(split_ratio):
            if i != len(split_ratio) - 1:
                num_split_i = int(split_ratio_i * self.num_edges)
                edges_split_i = shuffled_edge_indices[
                                split_offset:split_offset + num_split_i
                                ]
                split_offset += num_split_i
            else:
                edges_split_i = shuffled_edge_indices[split_offset:]

            if edges_split_i.numel() == 0:
                split_empty_flag = True
                split_offset = 0
                edges_split_list = []
                break
            edges_split_list.append(edges_split_i)

        if split_empty_flag:
            # perform `secure split` s.t. guarantees all splitted subgraph
            # contains at least one edge.
            for i, split_ratio_i in enumerate(split_ratio):
                if i != len(split_ratio) - 1:
                    num_split_i = 1 + int(
                        split_ratio_i * (self.num_edges - len(split_ratio))
                    )
                    edges_split_i = shuffled_edge_indices[
                                    split_offset:split_offset + num_split_i
                                    ]
                    split_offset += num_split_i
                else:
                    edges_split_i = shuffled_edge_indices[split_offset:]
                edges_split_list.append(edges_split_i)

        for edges_split_i in edges_split_list:
            # shallow copy all attributes
            graph_new = copy.copy(self)
            graph_new.edge_label_index = self.edge_index[:, edges_split_i]
            graph_new.edge_label = self.edge_label[edges_split_i]
            graph_new.edges_split = edges_split_i

            split_graphs.append(graph_new)
        return split_graphs