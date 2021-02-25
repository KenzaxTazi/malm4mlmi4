''' 
Different regressors
'''

import torch 
import torch.nn as nn 

__all__ = ['MLP']

class MLP(nn.Module):
    """MLP with 1 hidden layer
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """

    def __init__(self,in_channels,hidden_channels=[40,40],out_channels=1):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU()

        # Hidden layers
        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(self.in_channels, self.hidden_channels[0]))
        
        for k in range(len(self.hidden_channels)-1):
            self.hidden.append(nn.Linear(self.hidden_channels[k], self.hidden_channels[k+1]))

         # Output layer
        self.out = nn.Linear(self.hidden_channels[-1], self.out_channels)
        ####   

    def forward(self, x):
        
        #x = self.f(x)
        
        # Feedforward
        for layer in self.hidden[:]:
            x = self.relu(layer(x))
        x = self.out(x)
        return x