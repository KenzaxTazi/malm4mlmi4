import numpy as np
import matplotlib.pyplot as plt
import torch
import pdb

import os 
import data
import maml
import regressors

gen = data.SinusoidGenerator(batch_size = 16,
                 num_tasks = 16,
                 x_range = (-5, 5),
                 A_range = (0.1, 5),
                 P_range = (0, np.pi),
                 max_train_points=10,
                 max_test_points=10)

task = gen.generate_task()

# Create model and optimizer instances
model = regressors.MLP(in_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=.001)

# Create meta model instance
meta_model = maml.MAML_trainer(model, optimizer)

# Meta training loop
for idx, task in enumerate(gen):
    print(idx)
    meta_model.outer_loop_train(task['x_context'],
                          task['y_context'],
                          task['x_target'],
                          task['y_target'])

