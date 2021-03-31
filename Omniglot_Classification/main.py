
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
    meta_batch_size = 8
    conv = True
    epochs = 10

    steps = 3

    # Import data
    training_set, validation_set, test_set = dp.dataprep(meta_batch_size, K, N)
    xtrain_support, ytrain_support, xtrain_query, ytrain_query = training_set
    xval_support, yval_support, xval_query, yval_query = validation_set
    xtest_support, ytest_support, xtest_query, ytest_query = test_set

    # Create model and meta model instances
    classifier = m.Classifier(conv, K, N)
    meta_model = maml.MetaModel(classifier, lr, K, N)

    # Train and evaluate
    train_loss, train_acc  = meta_model.train(xtrain_support, ytrain_support, xtrain_query, ytrain_query, epochs=epochs)
    val_loss, val_acc  = meta_model.evaluate(xval_support, yval_support, xval_query, yval_query, steps)
    test_loss, test_acc  = meta_model.evaluate(xtest_support, ytest_support, xtest_query, ytest_query, steps)



