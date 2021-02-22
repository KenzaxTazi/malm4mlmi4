import torch
import torch.nn.functional as F
from regressors import *
import torch.nn as nn
import copy

class MAML_trainer():
    def __init__(self, regressor, optimizer):
        self.regressor = regressor
        self.optimizer = optimizer

    def _inner_loop_train(self, x_support, y_support, alpha):
        '''
        [TO DO: UPDATE]


        alpha = inner loop learning rate

        return a model copy with updated parameters
        '''

        # Forward pass using support sets
        predictions = self.regressor(x_support)
        loss = F.mse_loss(predictions, y_support)

        # Copy regressor to store updated params-> we don't want to update the actual meta-model
        copy_regressor = copy.deepcopy(self.regressor)
        copy_regressor_params = copy_regressor.named_parameters()

        # Manual backward pass
        for name, param in copy_regressor_params:
            #grad = param.grad.data
            grad = param.grad
            if grad is None:
                new_param = param
            else:
                new_param = param - alpha * grad.data # gradient descent
            #copy_regressor_params[name] = new_param
            param = new_param

        return copy_regressor

    def outer_loop_train(self, x_supports, y_supports, x_queries, y_queries, alpha=0.01, beta=0.01):
        '''

        [TO DO: UPDATE]


        Perform single outer loop forward and backward pass of MAML algorithm

        Structure:
        Support sets used for inner loop training
        Query sets used for outer loop training
        
        '''

        task_losses = []

        # Perform inner loop training per task using support sets
        for task in range(x_supports.size(0)):
            updated_regressor = self._inner_loop_train(x_supports[task], y_supports[task], alpha)

            # Collect predictions for query sets, using model updated params for specific task
            predictions = updated_regressor(x_queries[task])

            # Update task losses
            curr_loss = F.mse_loss(predictions, y_queries[task])
            task_losses.append(curr_loss)

        # Backward pass to update meta-model parameters
        total_loss = torch.stack(task_losses).sum()
        self.optimizer.zero_grad()
        total_loss.backward()
        for param in self.optimizer.param_groups[0]['params']:
            # Bit of regularisation innit
            nn.utils.clip_grad_value_(param, 10)
        self.optimizer.step()

        # Return training accuracy and loss
        loss = total_loss.item()
        # Need to use an AvergeMeter class to collect accuracies as we iterate through tasks
