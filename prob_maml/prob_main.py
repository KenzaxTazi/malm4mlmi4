import os
import numpy as np 
import torch
import torch.nn.functional as F
import copy
from collections import OrderedDict

from data_generator import DataGenerator
from model import ModelMLPSinusoid
from maml import MAML_trainer
from prob_model import ProbModelMLPSinusoid
from prob_maml import ProbMAML
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
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    meta_batch_size = 25
    update_batch_size = 5 # 10 for 2dclass
    num_samples_per_class = 55

    pretrain_iterations = 0
    metatrain_iterations = 70000 # 70000 in paper
    stochastic = True
    num_classes = 1 # 2 for 2dclass
    bias = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ProbModelMLPSinusoid(bias=bias).to(device)
    var_model = ProbModelMLPSinusoid(bias=bias, init_value=np.log(0.02 ** 2))
    optimizer = torch.optim.Adam(list(model.parameters()) + list(var_model.parameters()), lr=.01)
    meta_model = ProbMAML(model, optimizer, var_model)
    alpha=0.001
    num_inner_updates = 1

    data_generator = DataGenerator(num_samples_per_class, meta_batch_size, datasource=datasource, num_classes=num_classes)
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
        if idx % 5000 == 0:
            torch.save(meta_model.regressor.state_dict(), os.path.join(output_directory, f"{datasource}_model_{idx}.pt"))
            torch.save(meta_model.regressor_variances.state_dict(), os.path.join(output_directory, f"{datasource}_model_variances_{idx}.pt"))

    sns.set()
    plt.plot(losses, label='Loss')
    plt.plot(prior_losses, label='Prior Loss')
    plt.legend()

    plt.savefig(os.path.join(output_directory, f"{datasource}_losses.png"), dpi=300)
    torch.save(meta_model.regressor.state_dict(), os.path.join(output_directory, f"{datasource}_model.pt"))
    torch.save(meta_model.regressor_variances.state_dict(), os.path.join(output_directory, f"{datasource}_model_variances.pt"))

def test(datasource='sinusoid_linear', output_directory='prob_maml/results'):
    # meta_batch_size = 25
    bias = 0
    num_test_curves = 10
    num_samples_per_class = 55
    num_inner_updates = 5

    update_batch_size = 5 # 10 for 2dclass
    num_classes = 1 # 2 for 2dclass
    alpha=0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trained_regressor = ProbModelMLPSinusoid(bias=bias).to(device)
    trained_regressor.load_state_dict(torch.load(os.path.join(output_directory, f"{datasource}_model.pt")))

    trained_var_regressor = ProbModelMLPSinusoid(bias=bias).to(device)
    trained_var_regressor.load_state_dict(torch.load(os.path.join(output_directory, f"{datasource}_model_variances.pt")))
    
    optimizer = torch.optim.Adam(list(trained_regressor.parameters()) + list(trained_var_regressor.parameters()), lr=0.01)
    trained_meta_model = ProbMAML(trained_regressor, optimizer, trained_var_regressor)

    data_generator = DataGenerator(num_samples_per_class, num_test_curves, datasource=datasource, num_classes=num_classes)
    batch_x, batch_y, amp, phase = data_generator.generate(train=True, input_idx=update_batch_size)

    x_train = torch.tensor(batch_x[:, :update_batch_size, :].astype(np.float32)).to(device)
    x_test = torch.tensor(batch_x[:,update_batch_size:, :].astype(np.float32)).to(device)
    y_train = torch.tensor(batch_y[:, :update_batch_size, :].astype(np.float32)).to(device)
    y_test = torch.tensor(batch_y[:,update_batch_size:, :].astype(np.float32)).to(device)

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    sns.set()

    for idx, ax in enumerate(axes.flatten()):
        test_model = copy.deepcopy(trained_meta_model)
        updated_params = test_model._inner_loop_test(x_train[idx], y_train[idx], alpha)

        with torch.no_grad():
            prior_query_predictions = test_model.regressor(x_test[idx], OrderedDict(test_model.regressor.named_parameters()))
            prior_query_loss = F.mse_loss(prior_query_predictions, y_test[idx])

            updated_query_predictions = test_model.regressor(x_test[idx], updated_params)
            updated_query_loss = F.mse_loss(updated_query_predictions, y_test[idx])
            print(f"Prior loss: {prior_query_loss} | Adapted loss: {updated_query_loss}")

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
        ax.plot(x_train[idx], y_train[idx], '^', color='darkorchid', markeredgewidth=1, markersize=6, markeredgecolor='slategray', label='datapoints')
        ax.plot(x_test[idx], updated_query_predictions, color='red', linestyle='--', linewidth=1, label='MAML')

    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_directory, f"{datasource}_test.png"), dpi=300)


def test_both(datasource='sinusoid_linear', output_directory='prob_maml/results', og_maml_fname=None, prob_maml_fname=None, prob_maml_var_fname=None):
    # meta_batch_size = 25
    prob_test_trials = 5
    bias = 0
    num_test_curves = 10
    num_samples_per_class = 55
    num_inner_updates = 10

    update_batch_size = 5 # 10 for 2dclass
    num_classes = 1 # 2 for 2dclass
    alpha=0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_generator = DataGenerator(num_samples_per_class, num_test_curves, datasource=datasource, num_classes=num_classes)
    batch_x, batch_y, amp, phase = data_generator.generate(train=True, input_idx=update_batch_size)
    batch_x_ten, batch_y_ten, amp_ten, phase_ten = data_generator.generate(train=True, input_idx=10)

    x_train = torch.tensor(batch_x[:, :update_batch_size, :].astype(np.float32)).to(device)
    x_test = torch.tensor(batch_x[:,update_batch_size:, :].astype(np.float32)).to(device)
    y_train = torch.tensor(batch_y[:, :update_batch_size, :].astype(np.float32)).to(device)
    y_test = torch.tensor(batch_y[:,update_batch_size:, :].astype(np.float32)).to(device)

    x_train_ten = torch.tensor(batch_x_ten[:, :10, :].astype(np.float32)).to(device)
    x_test_ten = torch.tensor(batch_x_ten[:,10:, :].astype(np.float32)).to(device)
    y_train_ten = torch.tensor(batch_y_ten[:, :10, :].astype(np.float32)).to(device)
    y_test_ten = torch.tensor(batch_y_ten[:,10:, :].astype(np.float32)).to(device)

    trained_regressor_og = ModelMLPSinusoid(bias=bias).to(device)
    trained_regressor_og.load_state_dict(torch.load(og_maml_fname))
    optimizer_og = torch.optim.Adam(trained_regressor_og.parameters(), lr=0.0)
    trained_meta_model_og = MAML_trainer(trained_regressor_og, optimizer_og)

    trained_regressor = ProbModelMLPSinusoid(bias=bias).to(device)
    trained_regressor.load_state_dict(torch.load(prob_maml_fname))

    trained_var_regressor = ProbModelMLPSinusoid(bias=bias).to(device)
    trained_var_regressor.load_state_dict(torch.load(prob_maml_var_fname))
    
    optimizer = torch.optim.Adam(list(trained_regressor.parameters()) + list(trained_var_regressor.parameters()), lr=0.01)
    trained_meta_model = ProbMAML(trained_regressor, optimizer, trained_var_regressor)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    sns.set()

    for idx, ax in enumerate(axes.flatten()):
        if idx < 5:
            test_model_og = copy.deepcopy(trained_meta_model_og)
            test_model = copy.deepcopy(trained_meta_model)
            
            updated_params_og = OrderedDict(test_model_og.regressor.named_parameters())
            for update_idx in range(num_inner_updates):
                updated_params_og = test_model_og._inner_loop_test(x_train[idx], y_train[idx], alpha, updated_params_og)

            updated_params_trials = []
            for _ in range(prob_test_trials):
                updated_params = test_model._inner_loop_test(x_train[idx], y_train[idx], alpha)
                updated_params_trials.append(updated_params)

            with torch.no_grad():
                prior_query_predictions_og = test_model_og.regressor(x_test[idx], OrderedDict(test_model_og.regressor.named_parameters()))
                prior_query_loss_og = F.mse_loss(prior_query_predictions_og, y_test[idx])

                updated_query_predictions_og = test_model_og.regressor(x_test[idx], updated_params_og)
                updated_query_loss_og = F.mse_loss(updated_query_predictions_og, y_test[idx])
                print(f"OG MAML | Prior loss: {prior_query_loss_og} | Adapted loss: {updated_query_loss_og}")

                prior_query_predictions = test_model.regressor(x_test[idx], OrderedDict(test_model.regressor.named_parameters()))
                prior_query_loss = F.mse_loss(prior_query_predictions, y_test[idx])
                
                updated_query_predictions_trials = []
                for trial_idx in range(prob_test_trials):
                    updated_query_predictions = test_model.regressor(x_test[idx], updated_params_trials[trial_idx])
                    updated_query_loss = F.mse_loss(updated_query_predictions, y_test[idx])
                    updated_query_predictions_trials.append(updated_query_predictions)
                    print(f"Prob MAML {trial_idx} | Prior loss: {prior_query_loss} | Adapted loss: {updated_query_loss}")

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
            ax.plot(x_train[idx], y_train[idx], '^', color='darkorchid', markeredgewidth=1, markersize=6, markeredgecolor='black', label='datapoints')
            ax.plot(x_test[idx], updated_query_predictions_og, color='black', linestyle=':', linewidth=2, label='MAML')
            colors = ['red', 'orange', 'blue', 'green', 'pink']
            for trial_idx in range(prob_test_trials):
                ax.plot(x_test[idx], updated_query_predictions_trials[trial_idx], color=colors[trial_idx], linestyle='--', linewidth=1)
            
        if idx > 4:
            selected_idx = 5
            test_model_og = copy.deepcopy(trained_meta_model_og)
            test_model = copy.deepcopy(trained_meta_model)
            
            updated_params_og = OrderedDict(test_model_og.regressor.named_parameters())
            for update_idx in range(num_inner_updates):
                updated_params_og = test_model_og._inner_loop_test(x_train_ten[selected_idx][:idx], y_train_ten[selected_idx][:idx], alpha, updated_params_og)

            updated_params_trials = []
            for _ in range(prob_test_trials):
                updated_params = test_model._inner_loop_test(x_train_ten[selected_idx][:idx], y_train_ten[selected_idx][:idx], alpha)
                updated_params_trials.append(updated_params)

            with torch.no_grad():
                prior_query_predictions_og = test_model_og.regressor(x_test_ten[selected_idx], OrderedDict(test_model_og.regressor.named_parameters()))
                prior_query_loss_og = F.mse_loss(prior_query_predictions_og, y_test_ten[selected_idx])

                updated_query_predictions_og = test_model_og.regressor(x_test_ten[selected_idx], updated_params_og)
                updated_query_loss_og = F.mse_loss(updated_query_predictions_og, y_test_ten[selected_idx])
                print(f"OG MAML | Prior loss: {prior_query_loss_og} | Adapted loss: {updated_query_loss_og}")

                prior_query_predictions = test_model.regressor(x_test_ten[selected_idx], OrderedDict(test_model.regressor.named_parameters()))
                prior_query_loss = F.mse_loss(prior_query_predictions, y_test_ten[selected_idx])
                
                updated_query_predictions_trials = []
                for trial_idx in range(prob_test_trials):
                    updated_query_predictions = test_model.regressor(x_test_ten[selected_idx], updated_params_trials[trial_idx])
                    updated_query_loss = F.mse_loss(updated_query_predictions, y_test_ten[selected_idx])
                    updated_query_predictions_trials.append(updated_query_predictions)
                    print(f"Prob MAML {trial_idx} | Prior loss: {prior_query_loss} | Adapted loss: {updated_query_loss}")

            xs = np.arange(-5.0, 5.0, 0.01)
            if amp_ten[selected_idx] > 50:
                # linear func
                ys = (amp_ten[selected_idx]-100) * xs + (phase_ten[selected_idx]-100)
            else:
                # sine func
                ys = amp_ten[selected_idx] * np.sin(xs - phase_ten[selected_idx])

            ax.set_xlim([-5.0,5.0])
            ax.set_ylim([-9.0,9.0])
            ax.plot(xs, ys, '-',color='gray', linewidth=1, label='ground truth') 
            ax.plot(x_train_ten[selected_idx][:idx], y_train_ten[selected_idx][:idx], '^', color='darkorchid', markeredgewidth=1, markersize=6, markeredgecolor='black', label='datapoints')
            ax.plot(x_test_ten[selected_idx], updated_query_predictions_og, color='black', linestyle=':', linewidth=2, label='MAML')
            colors = ['red', 'orange', 'blue', 'green', 'pink']
            for trial_idx in range(prob_test_trials):
                ax.plot(x_test_ten[selected_idx], updated_query_predictions_trials[trial_idx], color=colors[trial_idx], linestyle='--', linewidth=1)
            

    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_directory, f"{datasource}_test_both.png"), dpi=300)

# Final MAML model: /modelbias6/
#test_both(datasource='sinusoid_linear', output_directory='prob_maml/results/prob_modelbias_6/', og_maml_fname='prob_maml/results/modelbias6/sinusoid_linear_model.pt')
#train(datasource='sinusoid_linear', output_directory='prob_maml/results/prob_modelbias_7/')
test_both(datasource='sinusoid_linear', output_directory='prob_maml/results/prob_modelbias_7/',
          og_maml_fname='prob_maml/results/modelbias6/sinusoid_linear_model.pt',
          prob_maml_fname='prob_maml/results/prob_modelbias_7/sinusoid_linear_model.pt',
          prob_maml_var_fname='prob_maml/results/prob_modelbias_7/sinusoid_linear_model_variances.pt')
