'''
Data preparation assumes the miniImageNet data has been downloaded and stored in .pkl files
miniImageNet data can be downloaded easily from Kaggle
'''
import pickle
import numpy as np
import torch
import random


def load(file_location, classes, class_size, img_size, channels):
    with open(file_location, "rb") as f:
        data = pickle.load(f)
    X = data["image_data"]
    X = X.reshape([classes, class_size, img_size, img_size, channels])
    X = torch.from_numpy(X)
    X = torch.transpose(torch.transpose(X, -1, -3), -1, -2)
    return X


def get_set(X_all, K, N_way):
    N_max = X.size(0)
    K_max = X.size(1)

    X_set = torch.zeros(K*N_way, X.size(-3), X.size(-2), X.size(-1))
    y_set = torch.zeros(K_shot*N_way)

    counter = 0
    for i in range(N_way):
        class_ind = random.randint(0, N_max-1)
        for j in range(K):
            img_ind = random.randint(0, K_max-1)
            X_select = X_all[class_ind][img_ind]
            X_set[counter] = X_select
            y_set[counter] = i
            counter += 1
    return X_set, y_set


def get_one_task_data(X_all, K_shot, K_query, N_way):

    X_support, y_support = get_set(X_all, K_shot, N_way)
    X_query, y_query = get_set(X_all, K_query, N_way)

    return X_support, y_support, X_query, y_query

def get_all_tasks(X_all, T, K_shot, K_query, N_way):

    X_support = torch.zeros(T, K_shot*N_way, X.size(-3), X.size(-2), X.size(-1))
    y_support = torch.zeros(T, K_shot*N_way)

    X_query = torch.zeros(T, K_query*N_way, X.size(-3), X.size(-2), X.size(-1))
    y_query = torch.zeros(T, K_query*N_way)

    for t in range(T):
        X_support_t, y_support_t, X_query_t, y_query_t = get_one_task_data(X_all, K_shot, K_query, N_way)
        X_support[t] = X_support_t
        y_support[t] = y_support_t
        X_query[t] = X_query_t
        y_query[t] = y_query_t

    return X_support, y_support, X_query, y_query

def get_train_data(T=600, K_shot=1, K_query=128, N_way=5):
    '''
    T is number of tasks to generate
    K is the number of example images per class
    N_way = number of classes per task
    '''
    file = './Data/mini-imagenet-cache-train.pkl'
    X_all = load(file, 64, 600, 84, 84, 3)

    return get_all_tasks(X_all, T, K_shot, K_query, N_way)

def get_val_data(T=600, K_shot=1, K_query=128, N_way=5):
    '''
    T is number of tasks to generate
    K is the number of example images per class
    N_way = number of classes per task
    '''
    file = './Data/mini-imagenet-cache-val.pkl'
    X_all = load(file, 16, 600, 84, 84, 3)

    return get_all_tasks(X_all, T, K_shot, K_query, N_way)

def get_test_data(T=600, K_shot=1, K_query=128, N_way=5):
    '''
    T is number of tasks to generate
    K is the number of example images per class
    N_way = number of classes per task
    '''
    file = './Data/mini-imagenet-cache-test.pkl'
    X_all = load(file, 20, 600, 84, 84, 3)

    return get_all_tasks(X_all, T, K_shot, K_query, N_way)
