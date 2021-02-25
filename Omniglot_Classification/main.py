
import model as m
import maml
import dataprep as dp
import torch
import torchvision


K = 1  # shots or number of updates
N = 5  # number of classes per task
update_lr = 0.4 
meta_batch_size = 32
conv = True

# Import data
xtrain, ytrain, xval, yval, xtest, ytest = dp.dataprep(K, N)
  
# Create model and meta model instances
classifier = m.Classifier(conv, K, N)
meta_model = maml.MetaModel(classifier, meta_batch_size, update_lr, K , N)

# Train and test
meta_model.meta_learn(xtrain, ytrain, xval, yval)



