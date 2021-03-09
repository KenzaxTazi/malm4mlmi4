
import model as m
import maml
import dataprep as dp
import torch
import torchvision


K = 1  # shots or number of updates
N = 5  # number of classes per task
update_lr = 0.4 
meta_batch_size = 10 #32
conv = True

# Import training data
xtrain_support, ytrain_support, xtrain_query, ytrain_query = dp.dataprep(meta_batch_size, K, N)
  
# Create model and meta model instances
classifier = m.Classifier(conv, K, N)
meta_model = maml.MetaModel(classifier, update_lr, K, N)

# Train and test
meta_model.meta_learn(xtrain_support, xtrain_query, ytrain_support, ytrain_query)



