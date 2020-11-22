"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """
        super(MLP, self).__init__()
        self.layers = []
        
        # check whether the network has hidden layers or not
        if len(n_hidden) >= 1:
          for n_outputs in n_hidden:
            self.layers.append(LinearModule(n_inputs,n_outputs))
            self.layers.append(ELUModule())
            n_inputs = n_outputs

        self.layers.append(LinearModule(n_inputs,n_classes))
        self.layers.append(SoftMaxModule())
        

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        for layer in self.layers:
          x = layer.forward(x)
        out = x
        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """
        # compute the gradient by backtracking through the network
        for layer in reversed(self.layers):
          dout = layer.backward(dout)

        return


    def updatableLayers(self):
      """
      Returns the layers of the network that have updateable parameters
      """
      for layer in self.layers:
        if isinstance(layer, LinearModule):
          yield layer