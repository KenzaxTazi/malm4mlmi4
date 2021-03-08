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

img_size = [28, 28]


def data_splitting():
    """ Split data """
    # Create character list
    char_list = glob('data/omniglot_resized/*/*')

    # Data splitting 
    np.random.shuffle(char_list)
    training_char = char_list[0:1200]
    validation_char = char_list [1200:1250]
    test_char = char_list[1250:-1]

    return  training_char, validation_char, test_char


def load_data(K, N, char_list):
    """ 
    Load images, augment and create onehot label. Returns:

    x = [T x K*N x C x H x W]
    y = [T x K*N]

    T: Number of tasks -> sometimes called episodes
    K: K-shot learning
    N: N-way i.e. number of classes for task
    C: Channels
    H: Image Height
    W: Image Width
    """

    ys = []
    xs = []
    rot = [0, 90, 180, 270]

    for folder in char_list:
        for i in range(4):

            x = []
            y = []

            for n in range(N):
    
                files = glob(os.path.join(folder, '*.png'))[0:K]

                for f in files:
                    # images
                    image = Image.open(f)
                    rot_image = image.rotate(rot[i])
                    image_arr = np.array(rot_image, dtype=int)
                    reshaped_image = image_arr.reshape(1, 28, 28)
                    x.append(reshaped_image)
                    

                    # label
                    filename =  os.path.basename(f)
                    index, _ = filename.split(sep='_')
                    label = np.zeros(1623*4)
                    label[int(index) + 1623*i -1] = 1
                    y.append(label)
                
            ys.append(y)
            xs.append(x)


    # To PyTorch tensors
    xs_tensor = torch.tensor(xs, dtype= torch.long)
    ys_tensor = torch.tensor(ys, dtype= torch.long)
    
    return xs_tensor, ys_tensor



def dataprep(K, N):
    
    training_char, validation_char, test_char = data_splitting()
    
    xtrain, ytrain = load_data(K, N, training_char)
    xval, yval = load_data(K, N, validation_char)
    xtest, ytest = load_data(K, N, test_char)

    return xtrain, ytrain, xval, yval, xtest, ytest


