import torch
import torch.nn.functional as F
from classifiers import *
import torch.nn as nn
from collections import OrderedDict

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])
def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

class MAML_trainer():
    def __init__(self, classifier, optimizer):
        self.classifier = classifier
        self.optimizer = optimizer

    def _inner_loop_train(self, x_support, y_support, alpha):
        '''
        x_support = [K*N x C x H x W]
        y_support = [K*N]

        K: K-shot learning i.e. number of examples of each image per class
        N: N-way i.e. number of classes for task
        C: Channels
        H: Image Height
        W: Image Width

        alpha = inner loop learning rate

        returns the updated parameters
        '''

        # Forward pass using support sets
        with torch.enable_grad():
            logits = self.classifier(x_support, OrderedDict(self.classifier.named_parameters()))
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
                    new_param = param - alpha * grad # gradient descent
                updated_params[name] = new_param

        #print(updated_params['classifier.weight'])
        return updated_params

    def outer_loop_train(self, x_supports, y_supports, x_queries, y_queries, alpha=0.01, beta=0.01):
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
        for task in range(x_supports.size(0)):
            updated_params = self._inner_loop_train(x_supports[task], y_supports[task], alpha)

            # Collect logit predictions for query sets, using updated params for specific task
            logits = self.classifier(x_queries[task], updated_params)

            # Update task losses
            curr_loss = F.cross_entropy(logits, y_queries[task])
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
