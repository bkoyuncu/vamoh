import torch.nn as nn

from imagegym.register import register_layer


class ExampleConv2(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(ExampleConv2, self).__init__()
        self.model = ExampleConv2Layer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


# Remember to register your layer!
register_layer('exampleconv2', ExampleConv2)
