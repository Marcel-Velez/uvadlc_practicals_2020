"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100


# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

# initialize the arrays to keep track of the loss and accuracy
test_overall_accuracy, train_overall_accuracy, test_overall_loss, train_overall_loss, test_x_axis, train_x_axis = [],[],[],[],[],[]

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """
    accuracy = (predictions.argmax(1) == targets.argmax(1)).sum().float().div(predictions.shape[0])
    
    return accuracy

def cifar_test_generator(dataset):
    """
    Creates a generator object that yields a test dataset in batches

    Args:
        dataset: cifar10 dataset object

    Returns:
        numpy array of samples
        numpy array of one hot encoded targets

    """    
    import math
    X, y = dataset['test'].images, dataset['test'].labels
    for i in range(math.ceil(len(X)/FLAGS.batch_size)):
        upper = (i+1)*FLAGS.batch_size 
        lower = i * FLAGS.batch_size
        try:
            yield X[lower:upper],y[lower:upper]
        except:
            yield X[lower:], y[lower:]

def train():
    """
    Performs training and evaluation of MLP model.
  
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    # select which device to train the model on
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # compute the input size of the MLP 
    input_size, n_classes = 3*32*32, 10
    
    # init model, define the dataset, loss function and optimizer
    model = MLP(input_size, dnn_hidden_units, n_classes, FLAGS.b).to(device)
    dataset = cifar10_utils.get_cifar10(FLAGS.data_dir)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate)

    for step in range(FLAGS.max_steps):
        X_train, y_train = dataset['train'].next_batch(FLAGS.batch_size)
        optimizer.zero_grad()

        # move to correct device and shape for MLP
        X_train, y_train = torch.tensor(X_train).reshape(FLAGS.batch_size, input_size).float().to(device), torch.tensor(y_train).float().to(device) 

        predictions = model(X_train)
        train_loss = loss_fn(predictions, y_train.argmax(1).long())
        
        train_loss.backward()
        optimizer.step()

        # add the loss and accuracy to the lists for plotting
        train_overall_loss.append(train_loss.cpu().detach().sum())
        train_overall_accuracy.append(accuracy(predictions.cpu().detach(), y_train.cpu().detach()))
        train_x_axis.append(step)

        # test the model when eval freq is reached or if it is the last step
        if not step % FLAGS.eval_freq or step+1 == FLAGS.max_steps:
            model.eval()
            test_accuracies, test_losses_list = [], []

            # test batchwise since it doesnot fit my gpu
            for X_test, y_test in cifar_test_generator(dataset):
                X_test, y_test = torch.tensor(X_test).reshape(FLAGS.batch_size, input_size).float().to(device), torch.tensor(y_test).float().to(device) 

                predictions =  model(X_test)
                test_loss = loss_fn(predictions, y_test.argmax(1).long())
                test_accuracy = accuracy(predictions, y_test)

                # add the values to compute the average loss and accuracy for the entire testset
                test_accuracies.append(test_accuracy.cpu().detach())
                test_losses_list.append(test_loss.cpu().detach().sum())


            print("[{:5}/{:5}] Train loss {:.5f} Test loss {:.5f} Test accuracy {:.5f}".format(
                step, FLAGS.max_steps, train_loss, test_loss, sum(test_accuracies)/len(test_accuracies)
            ))

            test_overall_accuracy.append( sum(test_accuracies)/len(test_accuracies))
            test_overall_loss.append(sum(test_losses_list)/len(test_losses_list))
            test_x_axis.append(step)

            model.train()

    plt.plot(train_x_axis, train_overall_loss, label="Avg Train loss")
    plt.plot( test_x_axis, test_overall_loss,label="Avg Test loss")
    plt.legend()
    plt.savefig("pytorch_loss_curve")
    plt.show()

    plt.plot( train_x_axis ,train_overall_accuracy, label="Train batch accuracy")
    plt.plot( test_x_axis,test_overall_accuracy, label="Test set accuracy")
    plt.legend()
    plt.savefig("pytorch_accuracy_curve")
    plt.show()


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('-b', type=bool, default=False,
                        help='boolean whether batchnorm is added after each ELU')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
