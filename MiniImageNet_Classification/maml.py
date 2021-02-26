import torch
import torch.nn.functional as F
from classifiers import *
import torch.nn as nn
import copy

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
    # TODO: NEED TO CHANGE STRUCTURE OF FORWARD FUNCTIONS TO ACCEPT UPDATED PARAMS AS TENSORS
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

        return a model copy with updated parameters
        '''

        # Forward pass using support sets
        with torch.enable_grad():
            logits = self.classifier(x_support)
            loss = F.cross_entropy(logits, y_support)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)

            # Copy classifier to store updated params-> we don't want to update the actual meta-model
            copy_classifier = copy.deepcopy(self.classifier)
            #copy_classifier_params = copy_classifier.state_dict()
            copy_classifier_params = copy_classifier.named_parameters()
            #meta_params = self.classifier.state_dict()
            #grads = {k:v.grad for k, v in zip(meta_params, self.classifier.parameters())}

            # Manual update
            for  (name, param) in self.classifier.named_parameters():
                grad = param.grad
                if grad is None:
                    new_param = param
                else:
                    new_param = param - alpha * grad # gradient descent

                del_attr(copy_classifier, name.split('.'))
                set_attr(copy_classifier, name.split('.'), new_param)
                #setattr(copy_classifier, name, new_param)
            #copy_classifier.load_state_dict(copy_classifier_params)
            #print(copy_classifier_params['network.fc.weight'])

        params = copy_classifier.state_dict()

        return copy_classifier

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
            updated_classifer = self._inner_loop_train(x_supports[task], y_supports[task], alpha)

            # Collect logit predictions for query sets, using model updated params for specific task
            logits = updated_classifer(x_queries[task])

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
