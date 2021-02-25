# Model

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Classifier(nn.Module):

    def __init__(self, conv, K, N):

        super().__init__()

        self.conv = conv
        self.bn = nn.BatchNorm2d(K*N)
        self.softmax = nn.Softmax(N)
        
        if self.conv == True:
            self.conv1 = nn.ConvTranspose2d(1, 64, 3) #3x3 convolutions 64 filters
            self.conv2 = nn.ConvTranspose2d(64, 64, 3)
            self.conv3 = nn.ConvTranspose2d(64, 64, 3)
            self.conv4 = nn.ConvTranspose2d(64, 64, 3) 
            self.fc1 = nn.Linear(64, 64)

        else:
            self.fc1 = nn.Linear(28*28, 256)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 64)
            self.fc4 = nn.Linear(64, 64)
            self.fc5 = nn.Linear(64, 64)


    def forward(self, x):
        '''
        x is [K*N x C x H x W] 
        K*N: batch size
        C = Channels
        H = Height
        W = Width
        '''

        if self.conv == True:
            x = F.relu(self.bn(self.conv1(x)))
            x = F.relu(self.bn(self.conv2(x)))
            x = F.relu(self.bn(self.conv3(x)))
            x = F.relu(self.bn(self.conv4(x)))
    
            x = self.softmax(self.fc1(x))

        else:
            x = F.relu(self.bn(self.fc1(x)))
            x = F.relu(self.bn(self.fc2(x)))
            x = F.relu(self.bn(self.fc3(x)))
            x = F.relu(self.bn(self.fc4(x)))
            x = self.sofmax(self.fc5(x))
            
        return x
