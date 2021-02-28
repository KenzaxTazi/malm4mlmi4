import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import pdb
import os 
import data
import maml
import regressors

from model import *
from utils import *

gen = data.SinusoidGenerator(batch_size = 16,
                 num_tasks = 16,
                 x_range = (-5, 5),
                 A_range = (0.1, 5),
                 P_range = (0, np.pi),
                 max_train_points=10,
                 max_test_points=10)

task = gen.generate_task()

# Create model and optimizer instances
#model = regressors.MLP(in_channels=1).
model = ModelMLPSinusoid()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=.01)

#model.train()
#optimizer.zero_grad()
#preds = model(task['x'])
#loss = F.mse_loss(preds, task['y'])
#loss.backward()
#optimizer.step()

# Create meta model instance
meta_model = maml.MAML_trainer(model, optimizer=optimizer)

# Meta training loop
for idx, task in enumerate(gen):
    print(idx)
    loss = meta_model.outer_loop_train(task['x_context'],
                          task['y_context'],
                          task['x_target'],
                          task['y_target'])
    
    print(idx, loss)

