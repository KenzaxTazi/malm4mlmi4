# Model

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchmeta.modules import MetaModule, MetaConv2d, MetaLinear, MetaBatchNorm2d


class Classifier(MetaModule):

    def __init__(self, conv, K, N):

        super(Classifier, self).__init__()

        self.conv = conv
        self.bn = MetaBatchNorm2d(64)
        
        if self.conv == True:
            self.conv1 = MetaConv2d(1, 64, 3, padding=1, stride=2) #3x3 convolutions 64 filters
            self.conv2 = MetaConv2d(64, 64, 3, padding=1, stride=2)
            self.conv3 = MetaConv2d(64, 64, 3, padding=1, stride=2)
            self.conv4 = MetaConv2d(64, 64, 3, padding=1, stride=2)
            self.fc1 = MetaLinear(64, N)

        else:
            self.fc1 = MetaLinear(28*28, 256)
            self.fc2 = MetaLinear(256, 128)
            self.fc3 = MetaLinear(128, 64)
            self.fc4 = MetaLinear(64, 64)
            self.fc5 = MetaLinear(64, 64)


    def forward(self, x, params=None):
        '''
        x is [K * N x C x H x W] 
        
        K = shots, class instances 
        N = number of classes for task
        C = Channels
        H = Height
        W = Width
        '''

        if self.conv == True:
            x = F.relu(self.bn(self.conv1(x, params=self.get_subdict(params, 'conv1')), params=self.get_subdict(params, 'bn')))
            x = F.relu(self.bn(self.conv2(x, params=self.get_subdict(params, 'conv2')), params=self.get_subdict(params, 'bn')))
            x = F.relu(self.bn(self.conv3(x, params=self.get_subdict(params, 'conv3')), params=self.get_subdict(params, 'bn')))
            x = F.relu(self.bn(self.conv4(x, params=self.get_subdict(params, 'conv4')), params=self.get_subdict(params, 'bn')))
            x = x.view((x.size(0), -1))  # reshape tensor 
            x = self.fc1(x, params=self.get_subdict(params, 'fc1'))

        else:
            x = F.relu(self.bn(self.fc1(x, params=self.get_subdict(params, 'fc1'))))
            x = F.relu(self.bn(self.fc2(x, params=self.get_subdict(params, 'fc2'))))
            x = F.relu(self.bn(self.fc3(x, params=self.get_subdict(params, 'fc3'))))
            x = F.relu(self.bn(self.fc4(x, params=self.get_subdict(params, 'fc4'))))
            x = x.view((x.size(0), -1))  # reshape tensor 
            x = self.fc5(x, params=self.get_subdict(params, 'fc5'))
            
        return x
