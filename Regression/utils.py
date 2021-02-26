import torch 

__all__ = ['device']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
