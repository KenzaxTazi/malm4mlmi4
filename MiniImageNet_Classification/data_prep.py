'''
Data preparation assumes the miniImageNet data has been downloaded and stored in .pkl files
miniImageNet data can be downloaded easily from Kaggle
'''
import pickle
import numpy as np
import torch

def load(file_location, classes, class_size, img_size, channels):
    with open(file_location, "rb") as f:
        data = pickle.load(f)
    X = data["image_data"]
    X = X.reshape([classes, class_size, img_size, img_size, channels])
    X = torch.from_numpy(X)
    X = torch.transpose(torch.transpose(X, -1, -3), -1, -2)
    return X

def get_one_task_data(N_shot, K):

def get_train_data(T=600, N_shot=1, K=5):
    '''
    T is number of tasks to generate
    N_shot is the number of example images in support set per class (also at test time)
    K = number of classes per task
    '''
    file = './Data/mini-imagenet-cache-train.pkl'
    X_all = load(file, 64, 600, 84, 84, 3)

def get_val_data():
    '''
    T is number of tasks to generate
    N_shot is the number of example images in support set per class (also at test time)
    K = number of classes per task
    '''

def get_test_data():
    '''
    T is number of tasks to generate
    N_shot is the number of example images in support set per class (also at test time)
    K = number of classes per task
    '''
