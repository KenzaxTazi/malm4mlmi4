import os
import numpy as np 
import torch
import torch.nn.functional as F
import copy
from collections import OrderedDict

from data_generator import DataGenerator
from model import ModelMLPSinusoid
from maml import MAML_trainer
from utils import RunningAverage

import seaborn as sns
import matplotlib.pyplot as plt

'''
Usage Instructions:
  Stochastic Experiments:
  --------------------------
  Sinusoid+linear regression, 5-shot:
      python3 main.py --datasource=sinusoid_linear --logdir=logs/stochastic_sinelin/learnmeanbeta_learnbothstd_evenlessclip --metatrain_iterations=70000 --update_batch_size=5 --stochastic=True --kl_weight=0.1 --num_updates=5 --dim_hidden=100 --context_var=20 --num_hidden=3 --inf_num_updates=1 --norm=None

  2D binary classification:
      python3 main.py --datasource=2dclass --logdir=logs/stochastic_2dclass/learnmeanbeta.bothstd --metatrain_iterations=70000 --update_batch_size=10 --stochastic=True --kl_weight=0.1 --num_updates=5 --dim_hidden=100 --context_var=20 --num_hidden=3 --inf_num_updates=1 --norm=batch_norm
'''

def train(datasource='sinusoid_linear', output_directory="prob_maml/results"):
    meta_batch_size = 25
    update_batch_size = 5 # 10 for 2dclass
    pretrain_iterations = 0
    metatrain_iterations = 30000 # 70000 in paper
    stochastic = True
    num_classes = 1 # 2 for 2dclass
    bias = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ModelMLPSinusoid(bias=bias).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    meta_model = MAML_trainer(model, optimizer)
    alpha=0.001
    num_inner_updates = 5

    data_generator = DataGenerator(update_batch_size + max(50, update_batch_size), meta_batch_size, datasource=datasource, update_batch_size=update_batch_size, num_classes=num_classes)
    num_classes = data_generator.num_classes

    losses = []
    prior_losses = []
    avg_loss = RunningAverage()
    avg_prior_loss = RunningAverage()

    for idx in range(pretrain_iterations + metatrain_iterations):
        batch_x, batch_y, amp, phase = data_generator.generate()

        # A for support, B for query

        inputa = torch.tensor(batch_x[:, :num_classes * update_batch_size, :].astype(np.float32)).to(device) # shape: meta_batch_size, update_batch_size, 
        labela = torch.tensor(batch_y[:, :num_classes * update_batch_size, :].astype(np.float32)).to(device)
        inputb = torch.tensor(batch_x[:, num_classes * update_batch_size:, :].astype(np.float32)).to(device) # b used for testing
        labelb = torch.tensor(batch_y[:, num_classes * update_batch_size:, :].astype(np.float32)).to(device)

        # The points are (x, y) = (input, label)
        loss, prior_loss = meta_model.outer_loop_train(inputa, labela, inputb, labelb, device, alpha, num_inner_updates)
        with torch.no_grad():
            avg_loss.update(loss)
            avg_prior_loss.update(prior_loss)
        losses.append(avg_loss.avg)
        prior_losses.append(avg_prior_loss.avg)

        print(f"Epoch: {idx} | Prior loss: {prior_loss} | Updated loss: {loss}")

    sns.set()
    plt.plot(losses, label='Loss')
    plt.plot(prior_losses, label='Prior Loss')
    plt.legend()
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    plt.savefig(os.path.join(output_directory, f"{datasource}_losses.png"), dpi=300)
    torch.save(meta_model.regressor.state_dict(), os.path.join(output_directory, f"{datasource}_model.pt"))

def test(datasource='sinusoid_linear', output_directory='prob_maml/results'):
    # meta_batch_size = 25
    bias = 0
    num_test_curves = 10
    num_samples_per_class = 105

    update_batch_size = 10 # 10 for 2dclass
    num_classes = 1 # 2 for 2dclass
    alpha=0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trained_regressor = ModelMLPSinusoid(bias=bias).to(device)
    trained_regressor.load_state_dict(torch.load(os.path.join(output_directory, f"{datasource}_model.pt")))
    optimizer = torch.optim.Adam(trained_regressor.parameters(), lr=.01)
    trained_meta_model = MAML_trainer(trained_regressor, optimizer)

    data_generator = DataGenerator(num_samples_per_class, num_test_curves, datasource=datasource, update_batch_size=update_batch_size, num_classes=num_classes)
    batch_x, batch_y, amp, phase = data_generator.generate(train=True, input_idx=update_batch_size)

    inputa = torch.tensor(batch_x[:, :update_batch_size, :].astype(np.float32)).to(device)
    inputb = torch.tensor(batch_x[:,update_batch_size:, :].astype(np.float32)).to(device)
    labela = torch.tensor(batch_y[:, :update_batch_size, :].astype(np.float32)).to(device)
    labelb = torch.tensor(batch_y[:,update_batch_size:, :].astype(np.float32)).to(device)

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    sns.set()
    for idx, ax in enumerate(axes.flatten()):
        test_model = copy.deepcopy(trained_meta_model)
        updated_params = test_model._inner_loop_train(inputa[idx], labela[idx], alpha)
        with torch.no_grad():
            prior_query_predictions = test_model.regressor(inputb[idx], OrderedDict(test_model.regressor.named_parameters()))
            prior_query_loss = F.mse_loss(prior_query_predictions, labelb[idx])

            updated_query_predictions = test_model.regressor(inputb[idx], updated_params)
            updated_query_loss = F.mse_loss(updated_query_predictions, labelb[idx])

        xs = np.arange(-5.0, 5.0, 0.01)
        if amp[idx] > 50:
            # linear func
            ys = (amp[idx]-100) * xs + (phase[idx]-100)
        else:
            # sine func
            ys = amp[idx] * np.sin(xs - phase[idx])

        ax.set_xlim([-5.0,5.0])
        ax.set_ylim([-9.0,9.0])
        ax.plot(xs, ys, '-',color='gray', linewidth=1, label='ground truth') 
        ax.plot(inputa[idx], labela[idx], '^', color='darkorchid', markeredgewidth=1, markersize=6, markeredgecolor='slategray', label='datapoints')
        ax.plot(inputb[idx], updated_query_predictions, color='red', linestyle='--', linewidth=1, label='MAML')

    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_directory, f"{datasource}_test.png"), dpi=300)

test(datasource='sinusoid_linear', output_directory='prob_maml/results/modelbias/')