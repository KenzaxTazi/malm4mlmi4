# MAML for classification 

import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from tools import *


class MetaModel():
    """ Create MAML instance for given model architecture """

    def __init__(self, classifier, update_lr, num_updates, num_classes):
        
        self.classifier = classifier
        self.update_lr = update_lr
        self.optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, betas=[0.9, 0.99])
    
    def inner_loop_train(self, x_support, y_support, steps):
        '''
        x_support = [K*N x C x H x W]
        y_support = [K*N]

        K: K-shot learning i.e. number of examples of each image per class
        N: N-way i.e. number of classes for task
        C: Channels
        H: Image Height
        W: Image Width

        return a model copy with updated parameters
        '''

        # Forward pass using support sets
        with torch.enable_grad():
            with torch.autograd.set_detect_anomaly(True):
                logits = self.classifier(x_support, OrderedDict(self.classifier.named_parameters()))
                y_support_indices = torch.argmax(y_support, dim=1)
                loss = F.cross_entropy(logits, y_support_indices)
                self.classifier.zero_grad()
                loss.backward(retain_graph=True)
                #print('Loss after support set', loss)

        updated_params = self.classifier.state_dict()

        # Manual update
        for  (name, param) in self.classifier.named_parameters():
            grad = param.grad
            
            if grad is None:
                new_param = param
            else:
                #print('why is this not changing? ', steps * self.update_lr * grad )
                new_param = param - steps * self.update_lr * grad  # gradient descent
            
            if name == 'conv2.weight':
                print('Old param ', param)
                print('New param ', new_param)
                print('Grad', grad)
            
            updated_params[name] = new_param

        return updated_params


    def meta_learn(self, x_supports, y_supports, x_queries, y_queries, epochs=1, train=True):
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
        accuracy = AverageMeter()

        for e in range(epochs):
            for batch in range(x_supports.size(0)):
                for task in range(x_supports.size(1)):

                    # Perform inner loop training per task using support sets
                    if train == True:
                        updated_params = self.inner_loop_train(x_supports[batch, task], y_supports[batch, task], steps=1)
                    else:
                        updated_params = self.inner_loop_train(x_supports[batch, task], y_supports[batch, task], steps=3)

                    # Collect logit predictions for query sets, using updated params for specific task
                    logits = self.classifier(x_queries[batch, task], updated_params)
                
                    # Calculate query task losses
                    y_queries_indices = torch.argmax(y_queries[batch, task], dim=1)
                    curr_loss = F.cross_entropy(logits, y_queries_indices)
                    #print('curr_loss: ', curr_loss)
                    total_loss = total_loss + curr_loss
                    
                    # Determine accuracy
                    acc = accuracy_topk(logits, y_queries_indices)
                    print('accuracy: ', acc)
                    accuracy.update(acc.item(), logits.size(0))

                #print('total_loss is: {} for batch number {}'.format(total_loss, batch))    

            # Backward pass to update meta-model parameters for each batch 
                if train: 
                    with torch.autograd.set_detect_anomaly(True):              
                        self.optimizer.zero_grad()
                        total_loss.backward()
                        for param in self.optimizer.param_groups[0]['params']:
                            # Bit of regularisation innit
                            nn.utils.clip_grad_value_(param, 10)
                        self.optimizer.step()
                        #self.zero_grad()

            # Return training accuracy and loss
            loss = total_loss.item()
            acc = accuracy.avg

        print('Final loss :', loss, 'Final acc :', acc)
        return loss, acc
