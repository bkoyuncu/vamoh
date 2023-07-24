from ast import Raise
import torch.nn as nn
import torch.nn.functional as F

from imagegym.config import cfg
from imagegym.models.act import act_dict

def process_cat_encoder_x_params(mlp_params, dims=None,coordinate_dim=None):
    assert 'layers' in mlp_params
    assert isinstance(mlp_params['layers'], list)
    if cfg.model.name_encoding == 'fourier':
        input_coor_dim = 2*cfg.params_encoding.num_frequencies
    else:
        input_coor_dim = coordinate_dim

    input_dim = input_coor_dim  ## encoded_coors_x -> projection of them which are smaller
    c_dim_list = [input_dim] + mlp_params['layers']
    mlp_params['c_dim_list'] = c_dim_list
    mlp_params['dims'] = dims

def process_cat_encoder_prior_params(mlp_params, dims=None, coordinate_dim=None):
    assert 'layers' in mlp_params
    assert isinstance(mlp_params['layers'], list)
 
    if cfg.model.name_encoding == 'fourier':
        input_coor_dim = 2*cfg.params_encoding.num_frequencies
    else:
        input_coor_dim = coordinate_dim


    if cfg.model.simple_cat ==True:
        if cfg.model.cat_encoder_x ==True:
            input_dim = cfg.params_cat_x["layers"][-1]
        else:
            input_dim = input_coor_dim  # p(c_d|x_d) 
    else:
        if cfg.model.cat_encoder_x ==True:
            input_dim = cfg.model.dim_z + cfg.params_cat_x["layers"][-1] # p(c_d|x_d_projected,z) 
        else:
            input_dim = cfg.model.dim_z + input_coor_dim  ## p(c_d|x_d,z) concats coordinates and z

    output_dim = cfg.params_k_mixture.K
    c_dim_list = [input_dim] + mlp_params['layers'] + [output_dim]
    mlp_params['c_dim_list'] = c_dim_list
    mlp_params['dims'] = dims


def process_cat_encoder_post_params(mlp_params, dims=None, coordinate_dim=None):
    assert 'layers' in mlp_params
    assert isinstance(mlp_params['layers'], list)
    #now we are also adding dim_z to posterior of the categorical

    if cfg.model.name_encoding == 'fourier':
        input_coor_dim = 2*cfg.params_encoding.num_frequencies
    else:
        input_coor_dim = coordinate_dim

    if cfg.model.post_cat_has_z==True:
        if cfg.model.simple_cat ==True:
            if cfg.model.cat_encoder_x ==True:
                input_dim = cfg.model.dim_z + cfg.params_cat_x["layers"][-1] #q(c_d|x_d_projected,z)
            else:
                input_dim = cfg.model.dim_z +  input_coor_dim # q(c_d|x_d,z)
        else:
            if cfg.model.cat_encoder_x ==True:
                input_dim = cfg.model.dim_z + cfg.dataset.dims[0] + cfg.params_cat_x["layers"][-1] #q(c_d|x_d_projected,y,z)
            else: 
                input_dim = cfg.model.dim_z + input_coor_dim + cfg.dataset.dims[0] # q(c_d|x_d,y,z)
    else:
        raise NotImplementedError
        # input_dim =  2*cfg.params_encoding.num_frequencies + cfg.dataset.dims[0] #concats coordinates and channels this does not use Z
    output_dim = cfg.params_k_mixture.K
    c_dim_list = [input_dim] + mlp_params['layers'] + [output_dim]
    mlp_params['c_dim_list'] = c_dim_list
    mlp_params['dims'] = dims



def basic_dense_block(in_dim, out_dim, activation, drop_rate=0.0, use_batch_norm=True, **kwargs):
    dense = nn.Linear(in_dim, out_dim)
    drop_layer = nn.Dropout(drop_rate, inplace=cfg.mem.inplace) if drop_rate > 0 else nn.Identity()
    if use_batch_norm:
        bn_layer = nn.BatchNorm1d(out_dim, eps=cfg.bn.eps, momentum=cfg.bn.mom)
    else:
        bn_layer = nn.Identity()
    return nn.Sequential(dense, bn_layer, activation, drop_layer)

#MLP identity
class Identity_layer(nn.Module):

    def __init__(self,
                 **kwargs):
        super(Identity_layer, self).__init__()
        #assert isinstance(c_dim_list, list)
        self.layers = [nn.Identity()]


    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # for layer in self.layers:
            # x = layer(x)
        logits = x
        return logits


class MLP_dropout(nn.Module):

    def __init__(self, c_dim_list,
                 act='prelu',
                 batchnorm=True,
                 dropout=0.0,
                 output_dropout=0.0,
                 l2norm=False,
                 dims=None,
                 **kwargs):
        super(MLP_dropout, self).__init__()
        assert isinstance(c_dim_list, list)
        n_layers = len(c_dim_list) - 1

        layers = []
        for i, (c_in, c_out) in enumerate(zip(c_dim_list[:-1], c_dim_list[1:])):
            if (i + 1) == n_layers:
                act_fn = nn.Tanh()
                drop_rate = output_dropout
            else:
                act_fn = act_dict[act]
                drop_rate = dropout

            layers.append(basic_dense_block(c_in, c_out,
                                            activation=act_fn,
                                            drop_rate=drop_rate,
                                            use_batch_norm=batchnorm,
                                            ))

        self.layers = nn.ModuleList(layers)

        self.has_l2norm = l2norm

        self.dims = dims

        self.output_dropout = output_dropout

    def set_output_dims(self, dims):
        self.dims = dims

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        for layer in self.layers:
            x = layer(x)
            if self.has_l2norm:
                x = F.normalize(x, p=2, dim=1)
        if self.dims is not None:
            logits = x.view(x.size(0), *self.dims)
        else:
            logits = x

        # if self.output_dropout != 0.0:
        #     dropout_layer = nn.Dropout(p=self.output_dropout)
        #     logits = dropout_layer(logits)
        return logits

class MLP(nn.Module):

    def __init__(self, c_dim_list,
                 act='prelu',
                 batchnorm=True,
                 dropout=0.0,
                 l2norm=False,
                 dims=None,
                 **kwargs):
        super(MLP, self).__init__()
        assert isinstance(c_dim_list, list)
        n_layers = len(c_dim_list) - 1

        layers = []
        for i, (c_in, c_out) in enumerate(zip(c_dim_list[:-1], c_dim_list[1:])):
            if (i + 1) == n_layers:
                act_fn = nn.Identity()
            else:
                act_fn = act_dict[act]

            layers.append(basic_dense_block(c_in, c_out,
                                            activation=act_fn,
                                            drop_rate=dropout,
                                            use_batch_norm=batchnorm,
                                            ))

        self.layers = nn.ModuleList(layers)

        self.has_l2norm = l2norm

        self.dims = dims

    def set_output_dims(self, dims):
        self.dims = dims

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        for layer in self.layers:
            x = layer(x)
            if self.has_l2norm:
                x = F.normalize(x, p=2, dim=1)
        if self.dims is not None:
            logits = x.view(x.size(0), *self.dims)
        else:
            logits = x

        return logits


    def stateless_forward(self, x, weights, biases):
        for i, layer in enumerate(self.layers):
            for j, module in enumerate(layer.children()):
                if j == 0:
                    x = F.linear(x, weights[i], biases[i])
                else:
                    x = module(x)
            if self.has_l2norm:
                x = F.normalize(x, p=2, dim=1)
        if self.dims is not None:
            logits = x.view(x.size(0), *self.dims)
        else:
            logits = x
        return logits

class MLP_simple(nn.Module):

    def __init__(self, c_dim_list,
                 act='prelu',
                 batchnorm=False,
                 dropout=0.0,
                 l2norm=False,
                 dims=None,
                 **kwargs):
        super(MLP_simple, self).__init__()
        assert isinstance(c_dim_list, list)
        n_layers = len(c_dim_list) - 1

        layers = []
        for i, (c_in, c_out) in enumerate(zip(c_dim_list[:-1], c_dim_list[1:])):
            if (i + 1) == n_layers:
                act_fn = nn.Identity()
            else:
                act_fn = act_dict[act]

            layers.append(basic_dense_block(c_in, c_out,
                                            activation=act_fn,
                                            drop_rate=dropout,
                                            use_batch_norm=batchnorm,
                                            ))

        self.layers = nn.ModuleList(layers)

        self.has_l2norm = l2norm

        self.dims = dims

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        logits = x
        return logits


def process_mlp_params(input_dim, output_dim, mlp_params, dims=None):
    assert 'dim_inner' in mlp_params
    assert 'num_layers' in mlp_params
    c_dim_list = [input_dim]
    dim_inner = mlp_params['dim_inner']
    layers_encoder = mlp_params['num_layers']
    c_dim_list.extend([dim_inner for i in range(layers_encoder)])
    c_dim_list.append(output_dim)
    mlp_params['c_dim_list'] = c_dim_list
    mlp_params['dims'] = dims