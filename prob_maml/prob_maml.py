
from collections import OrderedDict
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

from model import *


class ProbMAML():
    def __init__(self, regressor, optimizer, regressor_variances):
        self.regressor = regressor
        self.regressor_variances = regressor_variances
        self.optimizer = optimizer
        
    def _inner_loop_train(self, x_train, y_train, x_test, y_test, alpha=0.001, init_params=None, gamma_q=0.01, noise_q=0.01):
        '''
        Perform a single inner loop forward pass and backward pass (manual parameter update) of Model-Agnostic Meta Learning (MAML).  

        x_train : tensor [K, 1]
            Support set inputs, where K is the number of sampled input-output pairs from task.  

        y_train : tensor [K, 1]
            Support set outputs, where K is the number of sampled input-output pairs from task. 
        
        alpha : float
            Inner loop learning rate


        Return updated model parameters. 
        '''

        # Forward pass using support sets
        with torch.enable_grad():
            # How tf does an inner gradient step work with this??
            # if init_params is None:
            #     init_params =  OrderedDict(self.regressor.named_parameters())
            # predicted = self.regressor(x_test, init_params)
            predicted = self.regressor(x_test, OrderedDict(self.regressor.named_parameters()))
            loss = F.mse_loss(predicted, y_test)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)

            # mean_updated_params = self.regressor.state_dict()

            # # Manual update
            # for (name, param) in self.regressor.named_parameters():
            #     grad = param.grad
            #     if grad is None:
            #         new_param = param
            #     else:
            #         new_param = param - gamma_q * grad
            #     mean_updated_params[name] = new_param

            # sample_params = {}
            # q_means = {}
            # q_stds = {}
            # for name, dist_mean in mean_updated_params.items():
            #     dist_std = np.sqrt(noise_q) * torch.ones(param.shape)
            #     q_means[name] = dist_mean
            #     q_stds[name] = dist_std
            #     sample_params[name] = Normal(loc=dist_mean, scale=dist_std).sample().requires_grad_(True)
            

            # Sample weights from q_dist 
            sample_params = {}
            q_means = {}
            q_stds = {}

            for name, param in self.regressor.named_parameters():
                dist_mean = param - (gamma_q * param.grad)
                dist_std = np.sqrt(noise_q) * torch.ones(dist_mean.shape)
                q_means[name] = dist_mean
                q_stds[name] = dist_std
                sample_params[name] = Normal(loc=dist_mean, scale=dist_std).sample().requires_grad_(True)

            sample_predicted = self.regressor(x_train, OrderedDict(sample_params))
            sample_loss = F.mse_loss(sample_predicted, y_train)

            self.optimizer.zero_grad()
            sample_loss.backward(retain_graph=True)

            # updated_params = self.regressor.state_dict()
            updated_params = copy.deepcopy(sample_params)

            # Manual update
            for (name, param) in sample_params.items():
                grad = param.grad
                if grad is None:
                    new_param = param
                else:
                    new_param = param - alpha * grad
                updated_params[name] = new_param

        return updated_params, q_means, q_stds

    def outer_loop_train(self, x_trains, y_trains, x_tests, y_tests, device, alpha=0.001, num_inner_updates=5, gamma_p=0.01):
        '''
        Perform single outer loop forward and backward pass of Model-Agnostic Meta Learning Algorithm (MAML).

        x_trains : tensor [B, K, 1]
            Support sets inputs.  

        y_trains : tensor [B, K, 1]
            Support sets outputs. 

        x_tests : tensor [B, K, 1]
            Query sets inputs.  

        y_tests : tensor [B, K, 1]
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
        for task in range(x_trains.size(0)):
            updated_params = OrderedDict(self.regressor.named_parameters())
            for _ in range(num_inner_updates):
                updated_params, q_means, q_stds = self._inner_loop_train(x_trains[task], y_trains[task], x_tests[task], y_tests[task], alpha, updated_params)

            # Collect predictions for query sets, using model prior params for specific task
            prior_predictions = self.regressor(x_tests[task], OrderedDict(self.regressor.named_parameters()))

            # Calculate prior loss and add to task prior losses
            curr_prior_loss = F.mse_loss(prior_predictions, y_tests[task])
            total_prior_loss = total_prior_loss + curr_prior_loss

            # Collect predictions for query sets, using model updated params for specific task
            predictions = self.regressor(x_tests[task], updated_params)
            curr_loss = F.mse_loss(predictions, y_tests[task])

            total_loss = total_loss + curr_loss
            
        # Collect predictions for query sets, using model prior params for specific task
        final_prior_predictions = self.regressor(x_trains[task], OrderedDict(self.regressor.named_parameters()))

        # Calculate prior loss and add to task prior losses
        final_prior_loss = F.mse_loss(final_prior_predictions, y_trains[task])

        self.optimizer.zero_grad()
        final_prior_loss.backward()

        # Calculate p and q KL divergence
        p_all_means = []
        p_all_vars = []
        q_all_means = []
        q_all_stds = []

        for name, param in self.regressor.named_parameters():
            p_mean = param - (gamma_p * param.grad)
            p_var = self.regressor_variances.state_dict()[name]

            p_all_means.append(p_mean.flatten())
            p_all_vars.append(p_var.flatten())

            q_all_means.append(q_means[name].flatten())
            q_all_stds.append(q_stds[name].flatten())
    
        p_all_means = torch.cat(p_all_means)
        p_all_vars = torch.cat(p_all_vars)
        p_all_stds = torch.sqrt(torch.exp(p_all_vars))

        q_all_means = torch.cat(q_all_means)
        q_all_stds = torch.cat(q_all_stds)

        p_dist = Normal(loc=p_all_means, scale=p_all_stds)
        q_dist = Normal(loc=q_all_means, scale=q_all_stds)
        
        kl_div = torch.distributions.kl_divergence(q_dist, p_dist).mean()
        total_loss = total_loss + kl_div

        # Backward pass to update meta-model parameters
        self.optimizer.zero_grad()
        total_loss.backward()

        for param in self.optimizer.param_groups[0]['params']:
            # Bit of regularisation innit
            nn.utils.clip_grad_value_(param, 10)
        self.optimizer.step()

        # Return training loss
        return total_loss, total_prior_loss
    
    def _inner_loop_test(self, x_support, y_support, alpha):
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
    
    # def kl_divergence_gaussians(p, q):
    #     """Calculate KL divergence between 2 diagonal Gaussian
    #     Args: each paramter is list with 1st half as mean, and the 2nd half is log_std
    #     Returns: KL divergence
    #     """
    #     assert len(p) == len(q)

    #     n = len(p) // 2

    #     kl_div = 0
    #     for i in range(n):
    #         p_mean = p[i]
    #         p_log_std = p[n + i]

    #         q_mean = q[i]
    #         q_log_std = q[n + i]

    #         s1_vec = torch.exp(input=2 * q_log_std)
    #         mahalanobis = torch.sum(input=torch.square(input=p_mean - q_mean) / s1_vec)

    #         tr_s1inv_s0 = torch.sum(input=torch.exp(input=2 * (p_log_std - q_log_std)))

    #         log_det = 2 * torch.sum(input=q_log_std - p_log_std)

    #         kl_div_temp = mahalanobis + tr_s1inv_s0 + log_det - torch.numel(p_mean)
    #         kl_div_temp = kl_div_temp / 2

    #         kl_div = kl_div + kl_div_temp

    #     return kl_div