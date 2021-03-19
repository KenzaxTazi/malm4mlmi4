import torch
import torch.nn as nn 
import numpy as np

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaLinear, MetaSequential)

__all__ = ['ProbMetaMLPModel','ProbModelMLPSinusoid']

class ProbMetaMLPModel(MetaModule):
    """Multi-layer perceptron architecture.

    in_features : int 
        Number of input features.

    out_features : int 
        Number of output features.
    
    hidden_sizes : list in int
        Sizes of hidden layers.

    """

    def __init__(self, in_features, out_features, hidden_sizes, bias=None, init_value=None):
        super(ProbMetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
        self.bias = bias

        if self.bias is not None:
            self.in_features = in_features + self.bias

        layer_sizes = [self.in_features] + self.hidden_sizes

        def init_weights(m):
            if type(m) == nn.Linear:
                m.weight.data.fill_(init_value)
        
        self.features = MetaSequential(OrderedDict([('layer{0}'.format(i+1),
                        MetaSequential(OrderedDict([
                            ('linear', MetaLinear(hidden_size, layer_sizes[i+1], bias=True)),
                            ('relu', nn.ReLU())
                        ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.regressor = MetaLinear(hidden_sizes[-1], out_features, bias=True)

        if init_value is not None:
            self.features.apply(init_weights)
            self.regressor.weight.data.fill_(init_value)

        print(self)

    def forward(self, inputs, params=None):
        if self.bias is not None:
            target = torch.zeros(inputs.shape[0], inputs.shape[1] + self.bias)
            target[:, :inputs.shape[1]] = inputs
            inputs = target
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.regressor(features, params=self.get_subdict(params, 'regressor'))
        return logits

def ProbModelMLPSinusoid(hidden_sizes=[100, 100, 100], bias=None, init_value=None):
    return ProbMetaMLPModel(1, 1, hidden_sizes, bias, init_value)


