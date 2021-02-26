'''
Different classifier architectures for image classification
'''

import torch
import torch.nn as nn
from meta_model_blocks import *


class MetaMiniImageNetClassifier(MetaConvModel):
    def __init__(self, out_features, hidden_size=64):
        self.channels = 3
        super(MetaMiniImageNetClassifier, self).__init__(self.channels, out_features,
                                                            hidden_size=hidden_size,
                                                            feature_size=5*5*hidden_size)
