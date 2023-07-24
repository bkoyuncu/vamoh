
from imagegym.models.layer.mlp import MLP
from imagegym.contrib.layer import *
import imagegym.register as register


layer_dict = {
    'mlp': MLP,
}

# register additional convs
layer_dict = {**register.layer_dict, **layer_dict}
