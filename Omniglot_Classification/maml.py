# MAML for classification

import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict


class MetaModel():
    """
    Create MAML instance for given model architecture. To replicate paper, we need
    
    5-way, 1-shot omniglot:
        - metatrain_iterations=60000 
        - meta_batch_size=32
        - lr=0.4 
        - num_updates=1
        - logdir=logs/omniglot5way/
    20-way, 1-shot omniglot:
        - metatrain_iterations=60000 
        - meta_batch_size=16 
        - num_classes=20 
        - lr=0.1 
        - num_updates=5 
        - logdir=logs/omniglot20way/
    """


    def __init__(self, classifier, update_lr, num_updates, num_classes):
        
        self.classifier = classifier
        self.update_lr = update_lr
        self.num_updates = num_updates
        self.num_classes = num_classes

        self.optimizer = torch.optim.Adam(classifier.parameters(), lr=update_lr)

    
    def inner_loop_train(self, x_support, y_support):
        '''
        x_support = [K*N x C x H x W]
        y_support = [K*N]

        K: K-shot learning i.e. number of examples of each image per class
        N: N-way i.e. number of classes for task
        C: Channels
        H: Image Height
        W: Image Width

        alpha = inner loop learning rate

        return a model copy with updated parameters
        '''

        # Forward pass using support sets
        with torch.enable_grad():
            logits = self.classifier(x_support)
            loss = F.cross_entropy(logits, y_support)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)

            updated_params = self.classifier.state_dict()

            # Manual update
            for  (name, param) in self.classifier.named_parameters():
                grad = param.grad
                if grad is None:
                    new_param = param
                else:
                    new_param = param - self.update_lr * grad # gradient descent
                updated_params[name] = new_param

        #print(updated_params['classifier.weight'])
        return updated_params


    def meta_learn(self, x_supports, y_supports, x_queries, y_queries):
        '''
        Perform single outer loop forward and backward pass of MAML algorithm

        Structure:
        Support sets used for inner loop training
        Query sets used for outer loop training

        x_supports = [T x K_{support}*N x C x H x W], for inner loop training
        y_supports = [T x K_{support}*N], for inner loop training
        x_queries = [T x K_{query}*N x C x H x W], for outer loop update
        y_queries = [T x K_{query}*N], for outer loop update

        T: Number of tasks -> sometimes called episodes
        K: K-shot learning
        N: N-way i.e. number of classes for task
        C: Channels
        H: Image Height
        W: Image Width
        '''
  
        total_loss = torch.zeros(1)

        # Perform inner loop training per task using support sets
        for batch in range(x_supports.size(0)):
            for task in range(x_supports.size(1)):
                updated_params = self.inner_loop_train(x_supports[batch, task], y_supports[batch, task])

                # Collect logit predictions for query sets, using updated params for specific task
                logits = self.classifier(x_queries[batch, task], updated_params)

                # Update task losses
                curr_loss = F.cross_entropy(logits, y_queries[batch, task])
                total_loss = total_loss + curr_loss

            # Backward pass to update meta-model parameters
            self.optimizer.zero_grad()
            total_loss.backward()

        for param in self.optimizer.param_groups[0]['params']:
            # Bit of regularisation innit
            nn.utils.clip_grad_value_(param, 10)
        self.optimizer.step()

        # Return training accuracy and loss
        loss = total_loss.item()

        # Need to use an AvergeMeter class to collect accuracies as we iterate through tasks




