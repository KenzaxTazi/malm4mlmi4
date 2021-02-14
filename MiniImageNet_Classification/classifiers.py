'''
Different classifier architectures for image classification
'''

import torch
import torch.nn. as nn
from torchvision import models

class ResnetClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=False, trained_model=None):
        super().__init__()

        if pretrained:
            self.network=trained_model
            num_ftrs = self.network.fc.in_features
            self.network.network.fc = nn.Linear(num_ftrs, num_classes)
        else:
            self.network=models.resnet18(pretrained=False)
            self.network.avgpool = nn.AdaptiveAvgPool2d(1)
            num_ftrs = self.network.fc.in_features
            self.network.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, xb):
        '''
        xb is [N x C x H x W]
        N: batch size
        C = Channels
        H = Height
        W = Width
        '''
        return self.network(xb)
