"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

# for plotting the curves
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# for the transfer learning
import torchvision.models as models
from torchvision import transforms as T

#for debugging
import time
import ipdb

# define pretrained models dict
pretrained_dict = {
    "resnet18": models.resnet18,
    "alexnet" : models.alexnet,
    "squeezenet" : models.squeezenet1_0,
    "vgg16" : models.vgg16,
    "densenet" : models.densenet161,
    "inception" : models.inception_v3,
    "googlenet" : models.googlenet,
    "shufflenet" : models.shufflenet_v2_x1_0,
    "mobilenet" : models.mobilenet_v2,
    "resnext50_32x4d" : models.resnext50_32x4d,
    "wide_resnet50_2" : models.wide_resnet50_2,
    "mnasnet" : models.mnasnet1_0
    }

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
    Performs training and evaluation of ConvNet model.
  
    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    
    # select which device to train the model on
    device =  "cuda:0" if torch.cuda.is_available() else "cpu"
    # compute the input size of the MLP 
    input_size, n_classes = 3*32*32, 10
    
    # if no pretrained model is passed to the commandline the convnet model will be initialized
    if FLAGS.model == "custom":
        model = ConvNet(3, n_classes).to(device)
    else:
        # check if the requested pretrained is available
        assert FLAGS.model in pretrained_dict, "Model not available in pre_trained dict, given: {}, available: {}".format(FLAGS.model, pretrained_dict.keys())
        model = pretrained_dict[FLAGS.model](pretrained=True)
        
        # image transforms needed to be alligned with the type of input the pretrained model is used to
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        resize = T.Resize(256)
        centerCrop = T.CenterCrop(224)

        # break out the last layer and add a linear layer that has 10 outputs instead of 1000
        layers = []   
        first_linear = False    
        for child in model.children():
            if list(child.children()) == []:
                if isinstance(child, nn.Linear) and not first_linear:
                    layers.append(nn.Flatten())
                    first_linear = True
                layers.append(child)
            else:
                for grandchild in child.children():
                    if isinstance(grandchild, nn.Linear) and not first_linear:
                        layers.append(nn.Flatten())
                        first_linear = True
                    layers.append(grandchild)
        model = nn.Sequential(*layers[:-1], nn.Linear(in_features=layers[-1].in_features, out_features=10))
        model.to(device)

        # freeze the layers that are pretrained
        for child in list(model.children())[:-1]:
            for param in child.parameters():
                param.requires_grad = False

    # define the dataset, loss function and optimizer
    dataset = cifar10_utils.get_cifar10(FLAGS.data_dir)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FLAGS.learning_rate)

    for step in range(FLAGS.max_steps):
        X_train, y_train = dataset['train'].next_batch(FLAGS.batch_size)
        optimizer.zero_grad()

        # from to normalize the data for pretrained model https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122
        if FLAGS.model != "custom":
            X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float().to(device) 

            X_train -= X_train.min(1, keepdim=True)[0]
            X_train /= X_train.max(1, keepdim=True)[0]
            
            X_train = torch.tensor([normalize(T.ToTensor()(centerCrop(resize(T.ToPILImage()(x))))).numpy() for x in X_train]).to(device)
        else:
            # move to correct device and shape for MLP
            X_train, y_train = torch.tensor(X_train).float().to(device), torch.tensor(y_train).float().to(device) 

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
                if FLAGS.model != "custom":
                    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float().to(device) 

                    X_test -= X_test.min(1, keepdim=True)[0]
                    X_test /= X_test.max(1, keepdim=True)[0]
                    
                    X_test = torch.tensor([normalize(T.ToTensor()(centerCrop(resize(T.ToPILImage()(x))))).numpy() for x in X_test]).to(device)
                else:
                    # move to correct device and shape for MLP
                    X_test, y_test = torch.tensor(X_test).float().to(device), torch.tensor(y_test).float().to(device) 

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

            # freeze the pretrained layers
            if FLAGS.model != "custom":
                for child in list(model.children())[:-1]:
                    for param in child.parameters():
                        param.requires_grad = False

    plt.plot(train_x_axis, train_overall_loss, label="Avg Train loss")
    plt.plot( test_x_axis, test_overall_loss,label="Avg Test loss")
    plt.legend()
    plt.savefig("convnet_loss_curve")
    plt.show()

    plt.plot( train_x_axis ,train_overall_accuracy, label="Train batch accuracy")
    plt.plot( test_x_axis,test_overall_accuracy, label="Test set accuracy")
    plt.legend()
    plt.savefig("convnet_accuracy_curve")
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
    parser.add_argument('--model', type=str, default="custom",
                        help='if a pretrained model name is passed this will be finetuned on cifar10')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
