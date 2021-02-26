'''
Aim is to see if the parameters are updating
'''

import torch
from maml import MAML_trainer
from classifiers import ResnetClassifier

num_classes = 2
meta_classifier = ResnetClassifier(num_classes=num_classes)
meta_classifier.train()

lr = 10
optimizer = torch.optim.SGD(meta_classifier.parameters(), lr)
my_trainer = MAML_trainer(meta_classifier, optimizer)

# Create fake data
T = 3
K = 5
N = num_classes
C = 3
H = 10
W = 10

x_supports = torch.randn(T, K*N, C, H, W)
y_supports = (torch.rand(size=(T, K*N)) < 0.5).long()

x_queries = torch.randn(T, K*N, C, H, W)
y_queries = (torch.rand(size=(T, K*N)) < 0.5).long()

# perform a single outer loop train

params = meta_classifier.state_dict()
final_layer_before = params['network.fc.weight']
loss = my_trainer.outer_loop_train(x_supports, y_supports, x_queries, y_queries)
params = meta_classifier.state_dict()
final_layer_after = params['network.fc.weight']

print(final_layer_before)
print("------------------")
print(final_layer_after)
