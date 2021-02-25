'''
Aim is to see if the parameters are updating
'''

import torch
from maml import MAML_trainer
from classifiers import ResnetClassifier

num_classes = 2
meta_classifier = ResnetClassifier(num_classes=num_classes)
meta_classifier.train()

lr = 0.001
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
y_supports = (torch.rand(size=(T, K*N)) < 0.5).int()

x_queries = torch.randn(T, K*N, C, H, W)
y_queries = (torch.rand(size=(T, K*N)) < 0.5).int()

# perform a single outer loop train

final_layer_before = meta_classifier.network.fc
loss = my_trainer.outer_loop_train(self, x_supports, y_supports, x_queries, y_queries)
final_layer_after = meta_classifier.network.fc

print(final_layer_before)
print("------------------")
print(final_layer_after)
