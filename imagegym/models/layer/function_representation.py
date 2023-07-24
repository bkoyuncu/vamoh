import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from imagegym.models.layer.mlp import MLP_simple



def process_fnrep_params(encoding, feature_dim , fnrep_params):
    assert 'dim_inner' in fnrep_params
    assert 'num_layers' in fnrep_params
    c_dim_list = [encoding.feature_dim]
    dim_inner = fnrep_params['dim_inner']
    layers_encoder = fnrep_params['num_layers']
    c_dim_list.extend([dim_inner for i in range(layers_encoder)])
    c_dim_list.append(feature_dim)
    fnrep_params['c_dim_list'] = c_dim_list
    fnrep_params['encoding'] = encoding

class FunctionRepresentation(nn.Module):
    """Function to represent a single datapoint. For example this could be a
    function that takes pixel coordinates as input and returns RGB values, i.e.
    f(x, y) = (r, g, b).

    Args:
        coordinate_dim (int): Dimension of input (coordinates).
        feature_dim (int): Dimension of output (features).
        layer_sizes (tuple of ints): Specifies size of each hidden layer.
        encoding (torch.nn.Module): Encoding layer, usually one of
            Identity or FourierFeatures.
        final_non_linearity (torch.nn.Module): Final non linearity to use.
            Usually nn.Sigmoid() or nn.Tanh().
    """

    def __init__(self, c_dim_list,
                 encoding,
                 dropout=0.0,
                 act='prelu',
                 **kwargs):
        super(FunctionRepresentation, self).__init__()

        assert act not in ['prelu']
        self.c_dim_list = c_dim_list
        self.dropout = dropout
        self.act = act
        self.encoding = encoding

        if not isinstance(self.encoding, nn.Identity):
            assert self.encoding.feature_dim == c_dim_list[0]

        self.mlp = MLP_simple(c_dim_list,
                       act=act,
                       batchnorm=False,
                       dropout=dropout,
                       l2norm=False,
                       dims=None)

    def get_weight_shapes(self):
        """Returns lists of shapes of weights and biases in the network."""
        weight_shapes = []
        bias_shapes = []
        for param in self.mlp.parameters():
            if len(param.shape) == 1:
                bias_shapes.append(param.shape)
            if len(param.shape) == 2:
                weight_shapes.append(param.shape)
        return weight_shapes, bias_shapes

    def get_weights_and_biases(self):
        """Returns list of weights and biases in the network."""
        weights = []
        biases = []
        for param in self.mlp.parameters():
            if len(param.shape) == 1:
                biases.append(param)
            if len(param.shape) == 2:
                weights.append(param)
        return weights, biases

    def set_weights_and_biases(self, weights, biases):
        """Sets weights and biases in the network.

        Args:
            weights (list of torch.Tensor):
            biases (list of torch.Tensor):

        Notes:
            The inputs to this function should have the same form as the outputs
            of self.get_weights_and_biases.
        """
        weight_idx = 0
        bias_idx = 0
        with torch.no_grad():
            for param in self.mlp.parameters():
                if len(param.shape) == 1:
                    param.copy_(biases[bias_idx])
                    bias_idx += 1
                if len(param.shape) == 2:
                    param.copy_(weights[weight_idx])
                    weight_idx += 1

    def duplicate(self):
        """Returns a FunctionRepresentation instance with random weights."""
        # Extract device
        device = next(self.parameters()).device
        # Create new function representation and put it on same device
        return FunctionRepresentation(c_dim_list=self.c_dim_list,
                                      encoding=self.encoding,
                                      dropout=self.dropout,
                                      act=self.act).to(device)


    def batch_forward(self, coordinates, weights, biases):
        """Stateless forward pass for multiple function representations.

        Args:
            coordinates (torch.Tensor): Batch of coordinates of shape
                (batch_size, num_points, coordinate_dim).
            weights (dict of list of torch.Tensor): Batch of list of tensors
                containing weights of linear layers for each neural network.
            biases (dict of list of torch.Tensor): Batch of list of tensors
                containing biases of linear layers for each neural network.

        Return:
            Returns a tensor of shape (batch_size, num_points, feature_dim).
        """
        assert  len(coordinates.shape) == 3
        features = []
        bs = coordinates.shape[0]

        for i in range(bs):
            self.set_weights_and_biases(weights[i], biases[i])
            co = coordinates[i]
            features.append(
                self(co, weights[i], biases[i]).unsqueeze(0)
            )
        return torch.cat(features, dim=0)



    def stateless_forward(self, coordinates, weights, biases):
        """Computes forward pass of function representation given a set of
        weights and biases without using the state of the PyTorch module.

        Args:
            coordinates (torch.Tensor): Tensor of shape (num_points, coordinate_dim).
            weights (list of torch.Tensor): List of tensors containing weights
                of linear layers of neural network.
            biases (list of torch.Tensor): List of tensors containing biases of
                linear layers of neural network.

        Notes:
            This is useful for computing forward pass for a specific function
            representation (i.e. for a given set of weights and biases). However,
            it might be easiest to just change the weights of the network directly
            and then perform forward pass.
            Doing the current way is definitely more error prone because we have
            to mimic the forward pass, instead of just directly using it.

        Return:
            Returns a tensor of shape (num_points, feature_dim)
        """
        # Positional encoding is first layer of function representation
        # model, so apply this transformation to coordinates
        # hidden = self.encoding(coordinates)
        #INFO changed
        hidden = coordinates
        # Apply linear layers and non linearities
        hidden = self.mlp.stateless_forward(hidden, weights, biases)
        return hidden
    
    def position_encoding(self, coordinates):
        """Transform coordinates using the given position embedding
        Args:
            coordinates (torch.Tensor): Tensor of shape (num_points, coordinate_dim).
        Return:
            Returns a tensor of shape (num_points, feature_dim)
        """
        # Positional encoding is first layer of function representation
        # model, so apply this transformation to coordinates
        hidden = self.encoding(coordinates)
        return hidden





    def batch_stateless_forward(self, coordinates, weights, biases):
        """Stateless forward pass for multiple function representations.

        Args:
            coordinates (torch.Tensor): Batch of coordinates of shape
                (batch_size, num_points, coordinate_dim).
            weights (dict of list of torch.Tensor): Batch of list of tensors
                containing weights of linear layers for each neural network.
            biases (dict of list of torch.Tensor): Batch of list of tensors
                containing biases of linear layers for each neural network.

        Return:
            Returns a tensor of shape (batch_size, num_points, feature_dim).
        """
        assert  len(coordinates.shape) == 3
        features = []
        bs = coordinates.shape[0]

        for i in range(bs):
            co = coordinates[i] #torch.Size([784, 256]) already encoded
            features.append(
                self.stateless_forward(co, weights[i], biases[i]).unsqueeze(0)
            )
        return torch.cat(features, dim=0)  #torch.Size([96, 784, 1]) [bs,#points,ch]


class FourierFeatures(nn.Module):
    """Random Fourier features.

    Args:
        frequency_matrix (torch.Tensor): Matrix of frequencies to use
            for Fourier features. Shape (num_frequencies, num_coordinates).
            This is referred to as B in the paper.
        learnable_features (bool): If True, fourier features are learnable,
            otherwise they are fixed.
    """

    def __init__(self, frequency_matrix, learnable_features=False):
        super(FourierFeatures, self).__init__()
        if learnable_features:
            self.frequency_matrix = nn.Parameter(frequency_matrix)
        else:
            # Register buffer adds a key to the state dict of the model. This will
            # track the attribute without registering it as a learnable parameter.
            # We require this so frequency matrix will also be moved to GPU when
            # we call .to(device) on the model
            self.register_buffer('frequency_matrix', frequency_matrix)
        self.learnable_features = learnable_features
        self.num_frequencies = frequency_matrix.shape[0]
        self.coordinate_dim = frequency_matrix.shape[1]
        # Factor of 2 since we consider both a sine and cosine encoding
        self.feature_dim = 2 * self.num_frequencies

    def forward(self, coordinates):
        """Creates Fourier features from coordinates.

        Args:
            coordinates (torch.Tensor): Shape (num_points, coordinate_dim)
        """
        # The coordinates variable contains a batch of vectors of dimension
        # coordinate_dim. We want to perform a matrix multiply of each of these
        # vectors with the frequency matrix. I.e. given coordinates of
        # shape (num_points, coordinate_dim) we perform a matrix multiply by
        # the transposed frequency matrix of shape (coordinate_dim, num_frequencies)
        # to obtain an output of shape (num_points, num_frequencies).
        prefeatures = torch.matmul(coordinates, self.frequency_matrix.T)
        # Calculate cosine and sine features
        cos_features = torch.cos(2 * math.pi * prefeatures)
        sin_features = torch.sin(2 * math.pi * prefeatures)
        # Concatenate sine and cosine features
        return torch.cat((cos_features, sin_features), dim=1)




class IdentityFeatures(nn.Module):
    """Itendity features.

    Args:
        coordinate_dim (int): Dimension of the coordinates
    """

    def __init__(self, coordinate_dim):
        super(IdentityFeatures, self).__init__()
        self.feature_dim = coordinate_dim

    def forward(self, coordinates):
        """Creates Fourier features from coordinates.

        Args:
            coordinates (torch.Tensor): Shape (num_points, coordinate_dim)
        """
        return  coordinates
