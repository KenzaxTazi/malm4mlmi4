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



def load_data(batch_size, K, N, char_list, training_set=False):
    """ 
    Load images, augment and create onehot label. Returns:
        xs_support = [T x N x K x C x H x W]
        ys_support = [T x N x K x Z]
    
    and optionally:
        xs_query = [T x N x K_query x C x H x W]
        ys_query = [T x N x K_query x Z]

    T: Number of tasks -> sometimes called episodes
    K: K-shot learning
    N: N-way i.e. number of classes for task
    C: Channels
    H: Image Height
    W: Image Width
    Z: total number of labels (1623*4)
    """

    ys_support = []
    xs_support = []
    if training_set == True:
        xs_query = []
        ys_query = []

    rot = [0, 90, 180, 270]

    for folder in char_list:
        for i in range(4):

            x_classes_support = []
            y_classes_support = []

            if training_set == True:
                x_classes_query = []
                y_classes_query = []

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
                y_support = y_instances[:K]
                if training_set == True:
                    x_query = x_instances[K:]
                    y_query = y_instances[K:]

                y_classes_support.append(y_support)
                x_classes_support.append(x_support)
                if training_set == True:    
                    y_classes_query.append(y_query)
                    x_classes_query.append(x_query)

        ys_support.append(y_classes_support)
        xs_support.append(x_classes_support)   
        if training_set == True:  
            ys_query.append(y_classes_query)
            xs_query.append(x_classes_query)  

    xs_tensor, ys_tensor = shuffle_and_shape(xs_support, ys_support, batch_size)

    if training_set == True:
        xs_tensor_q, ys_tensor_q = shuffle_and_shape(xs_query, ys_query, batch_size)
        return xs_tensor, ys_tensor, xs_tensor_q, ys_tensor_q
    
    else:
        return xs_tensor, ys_tensor


def shuffle_and_shape(xs, ys, batch_size):
    """ 
    Return tensors with shape:
    x = [B x T x N*K x C x H x W]
    y = [B x T x N*K x Z]
    """
    # Shuffle
    np.random.shuffle(xs)
    np.random.shuffle(ys)

    # To Torch tensors
    xs_tensor = torch.tensor(xs, dtype= torch.float)
    ys_tensor = torch.tensor(ys, dtype= torch.long)

    # Reshape
    x_shp = xs_tensor.shape
    y_shp = ys_tensor.shape
    xs_batched = torch.reshape(xs_tensor, (-1, batch_size, x_shp[-5]*x_shp[-4], *x_shp[-3:]))
    ys_batched = torch.reshape(ys_tensor, (-1, batch_size, y_shp[-3]*y_shp[-2], y_shp[-1]))

    return xs_batched, ys_batched


def dataprep(batch_size, K, N):
    """
    Prepares the data for model. 
    Returns:
        xtrain_support, xtrain_query,
        ytrain_support, ytrain_query,
        xval, yval, xtest, ytest

    with shape:
    x or x_support = [B x T x N * K x 2 x C x H x W]
    y or y_support = [B x T x N * K x 2 x Z]
    x_query = [B x T x N * K_query x 2 x C x H x W]
    y_query = [B x T x N * K_query x 2 x Z]

    where:
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
    
    xtrain_support, xtrain_query, ytrain_support, ytrain_query = load_data(batch_size, K, N, training_char, training_set=True)
    #xval, yval = load_data(batch_size, K, N, validation_char)
    #xtest, ytest = load_data(batch_size, K, N, test_char)

    return xtrain_support, xtrain_query, ytrain_support, ytrain_query #xval, yval, xtest, ytest


