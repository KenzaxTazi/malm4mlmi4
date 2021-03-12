
import model as m
import maml
import dataprep as dp

"""
MAML Omniglot Experiment

    Convolutional 5-way, 1-shot omniglot:
        N = 5
        K = 1
        lr = 0.4
        meta_batch_size = 32
        conv= True
        meta_training_iterations = 60000
  
    Convolutional 20-way, 5-shot omniglot:
        N = 20 
        K = 5
        lr = 0.1
        meta_batch_size = 16
        conv= True
        meta_training_iterations = 60000         
"""

if __name__ == '__main__':
    K = 1
    N = 5
    lr = 0.4
    meta_batch_size = 1 #32
    conv = True
    meta_training_iterations = 60

    # Import training data
    xtrain_support, ytrain_support, xtrain_query, ytrain_query = dp.dataprep(meta_batch_size, K, N)
  
    # Create model and meta model instances
    classifier = m.Classifier(conv, K, N)
    meta_model = maml.MetaModel(classifier, lr, K, N)

    # Train and test
    meta_model.meta_learn(xtrain_support, ytrain_support, xtrain_query, ytrain_query, meta_learn_iterations=meta_training_iterations, train=True)



