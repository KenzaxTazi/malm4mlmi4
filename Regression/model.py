import torch.nn as nn 

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaLinear, MetaSequential)

Sinusoid

__all__ = ['MetaMLPModel','ModelMLPSinusoid']

class MetaMLPModel(MetaModule):
    """Multi-layer perceptron architecture.

    in_features : int 
        Number of input features.

    out_features : int 
        Number of output features.
    
    hidden_sizes : list in int
        Sizes of hidden layers.

    """

    def __init__(self, in_features, out_features, hidden_sizes):
        super(MetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        layer_sizes = [in_features] + hidden_sizes
        self.features = MetaSequential(OrderedDict([('layer{0}'.format(i+1),
                        MetaSequential(OrderedDict([
                            ('linear', MetaLinear(hidden_size, layer_sizes[i+1], bias=True)),
                            ('relu', nn.ReLU())
                        ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.classifier = MetaLinear(hidden_sizes[-1], out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits

def ModelMLPSinusoid(hidden_sizes=[40,40]):
    return MetaMLPModel(1, 1, hidden_sizes)

if __name__ == '__main__':
    print("model is MLPSinusoid")
    model = ModelMLPSinusoid()

