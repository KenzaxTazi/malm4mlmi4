# Data preperation

import numpy as np
import pandas as pd
import os
import random
import torch
import torchvision
from glob import glob
import random
from PIL import Image



"""
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
"""


def train_test_splitting():
    """ Split data into training, validation and test sets """
    # Create character list
    char_list = glob('data/omniglot_resized/*/*')

    # Data splitting 
    np.random.shuffle(char_list)
    training_char = char_list[0:120]
    validation_char = char_list [120:125]
    test_char = char_list[125:130]

    return  training_char, validation_char, test_char



def load_data(batch_size, K, N, char_list):
    """ 
    Load images, augment and create onehot label. Returns:

    x_tensor = [T x N x 2 x K x C x H x W]
    y_tensor = [T x N x 2 x K x Z]

    T: Number of tasks -> sometimes called episodes
    K: K-shot learning
    N: N-way i.e. number of classes for task
    C: Channels
    H: Image Height
    W: Image Width
    Z: total number of labels (1623*4)
    """

    ys = []
    xs = []
    rot = [0, 90, 180, 270]

    for folder in char_list:
        for i in range(4):

            x_classes= []
            y_classes= []

            for n in range(N):

                x_instances= []
                y_instances= []
    
                files = glob(os.path.join(folder, '*.png'))

                for f in files:
                    # images
                    image = Image.open(f)
                    rot_image = image.rotate(rot[i])
                    image_arr = np.array(rot_image, dtype=int)
                    reshaped_image = image_arr.reshape(1, 28, 28)
                    x_instances.append(reshaped_image)
                    
                    # label
                    filename =  os.path.basename(f)
                    index, _ = filename.split(sep='_')
                    label = np.zeros(1623*4)
                    label[int(index) + 1623*i -1] = 1
                    y_instances.append(label)

                x_support = x_instances[:K]
                x_query = x_instance[K:]
                y_support = y_instances[:K]
                y_query = y_instance[K:]
                
            ys.append([y_support, y_query])
            xs.append([x_support, x_query])

    xs_tensors, ys_tensor = shuffle_and_batch(xs, ys, batch_size, N, K)
    
    return xs_tensors, ys_tensor

def shuffle_and_batch(xs, ys, batch_size, N, K):
    """ 
    Return tensors with shape:
    x = [B x T x N x 2 x K x C x H x W]
    y = [B x T x N x 2 x K x Z]
    """
    # Shuffle
    np.random.shuffle(xs)
    np.random.shuffle(ys)

    # To Torch tensors
    xs_tensor = torch.tensor(xs, dtype= torch.float)
    ys_tensor = torch.tensor(ys, dtype= torch.long)

    # Reshape
    xs_batched = torch.reshape(xs_tensor, (-1, batch_size, N, 2, K, 1, 28, 28))
    ys_batched = ys_tensor.reshape((-1, batch_size, N, 2, K, 1623*4))

    return xs_batched, ys_batched


def dataprep(batch_size, K, N):
    """
    Prepares the data for model. 
    Returns:
        xtrain 
        ytrain 
        xval 
        yval 
        xtest
        ytest

    with shape:
    x = [B x T x N x 2 x K x C x H x W]
    y = [B x T x N x 2 x K x Z]

    B: Number of batches
    T: Number of tasks -> sometimes called episodes or batch size
    K: K-shot learning
    N: N-way i.e. number of classes for task
    C: Channels
    H: Image Height
    W: Image Width
    Z: total number of labels (1623*4)
    """
    
    training_char, validation_char, test_char = train_test_splitting()
    
    xtrain, ytrain, = load_data(batch_size, K, N, training_char)
    xval, yval = load_data(batch_size, K, N, validation_char)
    xtest, ytest = load_data(batch_size, K, N, test_char)

    return xtrain, ytrain, xval, yval, xtest, ytest


