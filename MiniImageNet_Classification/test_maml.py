'''
Aim is to see if the parameters are updating
'''

import torch
from maml import MAML_trainer
from classifiers import MetaMiniImageNetClassifier

seed = 1
torch.manual_seed(seed)

num_classes = 2
meta_classifier = MetaMiniImageNetClassifier(out_features=num_classes, hidden_size=64)
meta_classifier.train()

lr = 0.1
optimizer = torch.optim.SGD(meta_classifier.parameters(), lr)
my_trainer = MAML_trainer(meta_classifier, optimizer)

# Create fake data
T = 3
K = 5
N = num_classes
C = 3
H = 84
W = 84

x_supports = torch.randn(T, K*N, C, H, W)
y_supports = (torch.rand(size=(T, K*N)) < 0.5).long()

x_queries = torch.randn(T, K*N, C, H, W)
y_queries = (torch.rand(size=(T, K*N)) < 0.5).long()

# perform a single outer loop train
print("--------------------------------")
for name, param in meta_classifier.named_parameters():
    if name == 'classifier.weight':
        print(param.data)

loss = my_trainer.outer_loop_train(x_supports, y_supports, x_queries, y_queries)

print("--------------------------------")
for name, param in meta_classifier.named_parameters():
    if name == 'classifier.weight':
        print(param.data)
