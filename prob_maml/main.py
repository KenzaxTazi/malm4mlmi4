from data_generator import DataGenerator


'''
Usage Instructions:
  Stochastic Experiments:
  --------------------------
  Sinusoid+linear regression, 5-shot:
      python3 main.py --datasource=sinusoid_linear --logdir=logs/stochastic_sinelin/learnmeanbeta_learnbothstd_evenlessclip --metatrain_iterations=70000 --update_batch_size=5 --stochastic=True --kl_weight=0.1 --num_updates=5 --dim_hidden=100 --context_var=20 --num_hidden=3 --inf_num_updates=1 --norm=None

  2D binary classification:
      python3 main.py --datasource=2dclass --logdir=logs/stochastic_2dclass/learnmeanbeta.bothstd --metatrain_iterations=70000 --update_batch_size=10 --stochastic=True --kl_weight=0.1 --num_updates=5 --dim_hidden=100 --context_var=20 --num_hidden=3 --inf_num_updates=1 --norm=batch_norm
'''

datasource = 'sinusoid_linear' # or '2dclass_circle'
meta_batch_size = 25
update_batch_size = 5 # 10 for 2dclass
train = True
pretrain_iterations = 0
metatrain_iterations = 70000
stochastic = True
num_classes = 1 # 2 for 2dclass

if 'sinusoid' in datasource or '2dclass' in datasource:
    if train:
        test_num_updates = 5
    else:
        test_num_updates = 10

if train == False:
    orig_meta_batch_size = meta_batch_size
    # always use meta batch size of 1 when testing.
    meta_batch_size = 1


data_generator = DataGenerator(update_batch_size + max(50, update_batch_size), meta_batch_size, datasource=datasource, update_batch_size=update_batch_size, num_classes=num_classes)
num_classes = data_generator.num_classes

for idx in range(pretrain_iterations + metatrain_iterations):
    batch_x, batch_y, amp, phase = data_generator.generate(input_idx=idx)

    # A for support, B for query

    inputa = batch_x[:, :num_classes * update_batch_size, :] # shape: meta_batch_size, update_batch_size, 
    labela = batch_y[:, :num_classes * update_batch_size, :]
    inputb = batch_x[:, num_classes * update_batch_size:, :] # b used for testing
    labelb = batch_y[:, num_classes * update_batch_size:, :]

    # The points are (x, y) = (input, label)
    import pdb; pdb.set_trace()
