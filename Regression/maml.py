import torch
import torch.nn.functional as F
from model import *
import torch.nn as nn
from collections import OrderedDict
from utils import *

class MAML_trainer():
    def __init__(self, regressor, optimizer):
        self.regressor = regressor
        self.optimizer = optimizer
        
    def _inner_loop_train(self, x_support, y_support, alpha):
        '''
        Perform a single inner loop forward pass and backward pass (manual parameter update) of Model-Agnostic Meta Learning (MAML).  

        x_support : tensor [K, 1]
            Support set inputs, where K is the number of sampled input-output pairs from task.  

        y_support : tensor [K, 1]
            Support set outputs, where K is the number of sampled input-output pairs from task. 
        
        alpha : float
            Inner loop learning rate


        Return updated model parameters. 
        '''

        # Forward pass using support sets
        with torch.enable_grad():
            predicted = self.regressor(x_support, OrderedDict(self.regressor.named_parameters()))
            loss = F.mse_loss(predicted, y_support)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)

            updated_params = self.regressor.state_dict()

            # Manual update
            for (name, param) in self.regressor.named_parameters():
                grad = param.grad
                if grad is None:
                    new_param = param
                else:
                    new_param = param - alpha * grad
                updated_params[name] = new_param
        
        return updated_params

    def outer_loop_train(self, x_supports, y_supports, x_queries, y_queries, alpha=0.01):
        '''
        Perform single outer loop forward and backward pass of Model-Agnostic Meta Learning Algorithm (MAML).

        x_supports : tensor [B, K, 1]
            Support sets inputs.  

        y_supports : tensor [B, K, 1]
            Support sets outputs. 

        x_queries : tensor [B, K, 1]
            Query sets inputs.  

        y_queries : tensor [B, K, 1]
            Query sets outputs. 

            where:
                B : int : number of tasks per batch (i.e., batch size)
                K : int : number of sampled input-output pairs from task

        alpha : float
            Inner loop training rate. 
        
        Return:
            total_loss : tensor [1]
                batch loss with updated MAML model parmaters. 
            prior_total_loss : tensor [1]
                batch loss with prior MAML model parameters.   

        '''
        total_prior_loss = torch.zeros(1).to(device)
        total_loss = torch.zeros(1).to(device)

        # Perform inner loop training per task using support sets
        for task in range(x_supports.size(0)):
            updated_params = self._inner_loop_train(x_supports[task], y_supports[task], alpha)
            
            # Collect predictions for query sets, using model prior params for specific task
            prior_predictions = self.regressor(x_queries[task], OrderedDict(self.regressor.named_parameters()))

            # Calculate prior loss and add to task prior losses
            curr_prior_loss = F.mse_loss(prior_predictions, y_queries[task])
            total_prior_loss = total_prior_loss + curr_prior_loss

            # Collect predictions for query sets, using model updated params for specific task
            predictions = self.regressor(x_queries[task], updated_params)

            # Calculate updated losss and add to task updated losses
            curr_loss = F.mse_loss(predictions, y_queries[task])
            total_loss = total_loss + curr_loss

        # Backward pass to update meta-model parameters
        self.optimizer.zero_grad()
        total_loss.backward()

        for param in self.optimizer.param_groups[0]['params']:
            # Bit of regularisation innit
            nn.utils.clip_grad_value_(param, 10)
        self.optimizer.step()

        # Return training loss
        return total_loss, total_prior_loss