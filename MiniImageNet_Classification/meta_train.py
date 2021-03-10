'''
MAML meta-training specifically for MiniImageNet data
Can specify N-way, K_shot learning
'''

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import sys
import os
import argparse
from classifiers import MetaMiniImageNetClassifier
from data_prep import get_train_data, get_val_data
from maml import MAML_trainer
from tools import AverageMeter

def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        print("No CUDA found")
        return torch.device('cpu')

def train(train_loader, maml_trainer, epoch, device, lr, print_freq=1):
    '''
        Run one train epoch
    '''
    accs = AverageMeter()
    losses = AverageMeter()

    for i, (x_support, y_support, x_query, y_query) in enumerate(train_loader):

        x_support = x_support.to(device)
        y_support = y_support.to(device)
        x_query = x_query.to(device)
        y_query = y_query.to(device)

        loss, acc = maml_trainer.train(x_support, y_support, x_query, y_query, alpha=lr)

        losses.update(loss, x_support.size(0))
        accs.update(acc, x_support.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), loss=losses))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), acc=accs))

def validate(val_loader, maml_trainer, device, lr):
    '''
        Run evaluation
    '''
    accs = AverageMeter()
    losses = AverageMeter()
    for i, (x_support, y_support, x_query, y_query) in enumerate(val_loader):

        x_support = x_support.to(device)
        y_support = y_support.to(device)
        x_query = x_query.to(device)
        y_query = y_query.to(device)

        loss, acc = maml_trainer.evaluate(x_support, y_support, x_query, y_query, alpha=lr)
        losses.update(loss, x_support.size(0))
        accs.update(acc, x_support.size(0))

    print('Test\t  Loss: {loss.avg:.3f}\n'
          .format(loss=losses))
    print('Test\t  Accuracy: {acc.avg:.3f}\n'
          .format(acc=accs))

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('OUT', type=str, help='Specify output th file')
    commandLineParser.add_argument('--N', type=int, default=5, help="Specify N_way classification")
    commandLineParser.add_argument('--K', type=int, default=1, help="Specify K_shot learning")
    commandLineParser.add_argument('--B', type=int, default=4, help="Specify task batch size")
    commandLineParser.add_argument('--T', type=int, default=600, help="Specify number of training tasks")
    commandLineParser.add_argument('--seed', type=int, default=1, help='Specify seed')

    args = commandLineParser.parse_args()
    out_file = args.OUT
    N = args.N
    K = args.K
    B = args.B
    T = args.T
    seed = args.seed
    torch.manual(seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/meta_training.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    device = get_default_device()

    # Load the data as tensors
    X_support_train, y_support_train, X_query_train, y_query_train = get_train_data(T=T, K_shot=K, K_query=600-K, N_way=N)
    X_support_val, y_support_val, X_query_val, y_query_val = get_val_data(T=int(0.1*T), K_shot=K, K_query=600-K, N_way=N)

    # Use dataloader to handle batches easily
    train_ds = TensorDataset(X_support_train, y_support_train, X_query_train, y_query_train)
    val_ds = TensorDataset(X_support_val, y_support_val, X_query_val, y_query_val)

    train_dl = DataLoader(train_ds, batch_size=B, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=B, shuffle=False)

    # Initialise meta_classifier
    meta_model = MetaMiniImageNetClassifier(N)
    meta_model.to(device)

    # Define training hyperparams
    lr = 0.001
    epochs = 50
    sch = 0.98

    optimizer = torch.optim.SGD(meta_model.parameters(), lr=lr, momentum = 0.9, nesterov=True)

    # Initialise MAML trainer
    maml_trainer = MAML_trainer(meta_model, optimizer)

    for epoch in range(epochs):
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_dl, maml_trainer, epoch, device, lr)
        scheduler.step()

        # Evaluate on valdation set
        validate(val_dl, maml_trainer, device, lr)


    # Save the meta-model
    state = model.state_dict()
    torch.save(state, out_file)
