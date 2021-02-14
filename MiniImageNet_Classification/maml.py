import torch
import torch.nn.functional as F
from classifiers import *

class MAML_trainer():
    def __init__(self, classifier):
        self.classifier = classifier

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

    def outer_loop_train(self, x_supports, y_supports):
        '''
        Perform single outer loop forward and backward pass of MAML algorithm

        Structure:
        Support sets used for inner loop training
        Query sets used for outer loop training

        x_supports = [T x K_{support}*N x C x H x W], for inner loop training
        y_supports = [T x K_{support}*N], for inner loop training
        x_queries = [T x K_{query}*N x C x H x W], for measuring inner loop trained model performance ->

        T: Number of tasks -> sometimes called episodes
        K: K-shot learning
        N: N-way i.e. number of classes for task
        C: Channels
        H: Image Height
        W: Image Width
        '''

        # Perform inner loop training per task using support sets

        # Collect logit predictions for query sets, using updated params from inner loop

        # Calculate an overall loss

        # Backward pass to update meta-model parameters
