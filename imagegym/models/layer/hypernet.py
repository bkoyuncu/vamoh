# Based on https://github.com/EmilienDupont/neural-function-distributions
import torch
import torch.nn as nn
from imagegym.models.layer.mlp import MLP

def process_hypernet_params(input_dim, fn_representation, hyper_params):
    assert 'dim_inner' in hyper_params
    assert 'num_layers' in hyper_params
    assert 'coords_dim' in hyper_params
    assert hyper_params["coords_dim"] == 0 

    c_dim_list = [input_dim]
    dim_inner = hyper_params['dim_inner']
    layers_encoder = hyper_params['num_layers']
    c_dim_list.extend([dim_inner*(i+1) for i in range(layers_encoder)])
    hyper_params['c_dim_list'] = c_dim_list
    hyper_params['function_representation'] = fn_representation


class HyperNetwork(nn.Module):
    """Hypernetwork that outputs the weights of a function representation.

    Args:
        function_representation (models.function_representation.FunctionRepresentation):
        latent_dim (int): Dimension of latent vectors.
        layer_sizes (tuple of ints): Specifies size of each hidden layer.
        non_linearity (torch.nn.Module):
    """
    def __init__(self, function_representation,
                 c_dim_list,
                 dropout=0.0,
                 act='prelu',
                 coords_dim=0,
                 **kwargs):
        super(HyperNetwork, self).__init__()
        assert function_representation is not None
        assert isinstance(c_dim_list, list)
        self.function_representation = function_representation
        self.c_dim_list = c_dim_list
        self.dropout = dropout
        self.act = act
        self.coords_dim = coords_dim
        self.mlp = None
        self._infer_output_shapes()
        self._init_neural_net()

    def _infer_output_shapes(self):
        """Uses function representation to infer correct output shapes for
        hypernetwork (i.e. so dimension matches size of weights in function
        representation) network."""
        self.weight_shapes, self.bias_shapes = self.function_representation.get_weight_shapes()
        num_layers = len(self.weight_shapes)

        # Calculate output dimension
        self.output_dim = 0
        for i in range(num_layers):
            # Add total number of weights in weight matrix
            self.output_dim += self.weight_shapes[i][0] * self.weight_shapes[i][1]
            # Add total number of weights in bias vector
            self.output_dim += self.bias_shapes[i][0]

        # Calculate partition of output of network, so that output network can
        # be reshaped into weights of the function representation network
        # Partition first part of output into weight matrices
        start_index = 0
        self.weight_partition = []
        for i in range(num_layers):
            weight_size = self.weight_shapes[i][0] * self.weight_shapes[i][1]
            self.weight_partition.append((start_index, start_index + weight_size))
            start_index += weight_size

        # Partition second part of output into bias matrices
        self.bias_partition = []
        for i in range(num_layers):
            bias_size = self.bias_shapes[i][0]
            self.bias_partition.append((start_index, start_index + bias_size))
            start_index += bias_size

        self.num_layers_function_representation = num_layers

    def _init_neural_net(self):
        """Initializes weights of hypernetwork."""

        c_dim_list = [*self.c_dim_list, self.output_dim]
        if self.coords_dim > 0:
            c_dim_list[0] += self.coords_dim
        self.mlp = MLP(c_dim_list=c_dim_list,
                       act=self.act,
                       batchnorm=False,
                       dropout=self.dropout,
                       l2norm=False,
                       dims=None)
    def output_to_weights(self, output):
        """Converts output of function distribution network into list of weights
        and biases for function representation networks.

        Args:
            output (torch.Tensor): Output of neural network as a tensor of shape
                (batch_size, self.output_dim).

        Notes:
            Each element in batch will correspond to a separate function
            representation network, therefore there will be batch_size sets of
            weights and biases.
        """
        all_weights = {}
        all_biases = {}
        # Compute weights and biases separately for each element in batch
        for i in range(output.shape[0]):
            weights = []
            biases = []
            # Add weight matrices
            for j, (start_index, end_index) in enumerate(self.weight_partition):
                weight = output[i, start_index:end_index]
                weights.append(weight.view(*self.weight_shapes[j]))
            # Add bias vectors
            for j, (start_index, end_index) in enumerate(self.bias_partition):
                bias = output[i, start_index:end_index]
                biases.append(bias.view(*self.bias_shapes[j]))
            # Add weights and biases for this function representation to batch
            all_weights[i] = weights
            all_biases[i] = biases
        return all_weights, all_biases

    def output_to_weights_2(self, output):
        """Converts output of function distribution network into list of weights
        and biases for function representation networks.

        Args:
            output (torch.Tensor): Output of neural network as a tensor of shape
                (batch_size, self.output_dim).

        Notes:
            Each element in batch will correspond to a separate function
            representation network, therefore there will be batch_size sets of
            weights and biases.
        """
        # Compute weights and biases separately for each element in batch
        bs = output.shape[0]
        weights = []
        biases = []
        ws_and_bs= []

        for j in range(len(self.weight_partition)):
            start_index, end_index = self.weight_partition[j]
            weight = output[:, start_index:end_index]
            ws_and_bs.append(weight.reshape((bs,*self.weight_shapes[j])))
            start_index, end_index = self.bias_partition[j]
            bias = output[:, start_index:end_index]
            ws_and_bs.append(bias.reshape((bs,*self.bias_shapes[j])))
        return ws_and_bs
    
    def forward(self, latents, coords=None):
        """Compute weights of function representations from latent vectors.

        Args:
            latents (torch.Tensor): Shape (batch_size, latent_dim).
            coord (torch.Tensor): Shape (batch_size, coord_dim).
        """
        # if self.coords_dim > 0:
        #     # print("input z and coordinates are concat")
        #     input_ = torch.cat((latents, coords), dim=1)

        
        input_ = latents
        
        output = self.mlp(input_) # 
        # output_reshaped = self.output_to_weights(output)
        return output