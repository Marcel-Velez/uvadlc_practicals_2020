"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# for the loss and accuracy curves
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
train_losses, test_losses, train_accuracies, test_accuracies, train_x_axis, test_x_axis = [],[],[],[],[],[]

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
    accuracy = (predictions.argmax(1) == targets.argmax(1)).sum()/predictions.shape[0]

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

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    # compute the input size of the MLP 
    input_size, n_classes = 3*32*32, 10
    
    # init model, define the dataset, loss function and optimizer
    model = MLP(input_size, dnn_hidden_units, n_classes)
    dataset = cifar10_utils.get_cifar10(FLAGS.data_dir)
    loss_fn = CrossEntropyModule()

    for step in range(FLAGS.max_steps):
        # get next batch and compute loss
        X_train, y_train = dataset['train'].next_batch(FLAGS.batch_size)
        X_train = X_train.reshape(X_train.shape[0],input_size)
        predictions = model.forward(X_train)
        train_loss = loss_fn.forward(predictions, y_train)

        # backprop loss
        model.backward(loss_fn.backward(predictions,y_train))

        # add the loss and accuracy to the lists for plotting
        train_losses.append(train_loss.sum())
        train_accuracies.append(accuracy(predictions, y_train))
        train_x_axis.append(step)

        # update weigths
        for layer in model.updatableLayers():
            layer.params['weight'] -= (FLAGS.learning_rate * layer.grads['weight'] )
            layer.params['bias'] -= (FLAGS.learning_rate * layer.grads['bias'] )

        # test the model when eval freq is reached or if it is the last step
        if not step % FLAGS.eval_freq:
            test_accuracies_list, test_losses_list = [],[]

            # test batchwise since it doesnot fit my gpu
            for X_test, y_test in cifar_test_generator(dataset):

                X_test, y_test = X_test.reshape(X_test.shape[0], input_size), y_test 
                predictions =  model.forward(X_test)

                test_loss = loss_fn.forward(predictions, y_test)
                test_accuracy = accuracy(predictions, y_test)

                # add the values to compute the average loss and accuracy for the entire testset
                test_accuracies_list.append(test_accuracy)
                test_losses_list.append(test_loss.sum())

            print("[{:5}/{:5}] Train loss {:.5f} Test loss {:.5f} Test accuracy {:.5f}".format(
                step, FLAGS.max_steps, train_loss.sum(), test_loss.sum(), (sum(test_accuracies_list)/len(test_accuracies_list))
            ))

            test_losses.append(sum(test_losses_list)/len(test_losses_list))
            test_accuracies.append(sum(test_accuracies_list)/len(test_accuracies_list))
            test_x_axis.append(step)

    plt.plot(train_x_axis, train_losses, label="Avg Train loss")
    plt.plot( test_x_axis, test_losses,label="Avg Test loss")
    plt.legend()
    plt.savefig("mlp_numpy_loss_curve")
    plt.show()

    plt.plot( train_x_axis ,train_accuracies, label="Train batch accuracy")
    plt.plot( test_x_axis,test_accuracies, label="Test set accuracy")
    plt.legend()
    plt.savefig("mlp_numpy_accuracy_curve")
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
    FLAGS, unparsed = parser.parse_known_args()

    main()
