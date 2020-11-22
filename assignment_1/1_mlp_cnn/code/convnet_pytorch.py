"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class PreActModule(nn.Module):
  """
  This class implements a Pre activation module in PyTorch.
  Once initialized a pre activation object can perform forward.
  """
  def __init__(self, in_chan, out_chan, stride, pad, kern_size):
    """
    Initializes PreActModule object.
    
    Args:
      in_chan: number of input channels
      out_chan: number of output channels
      stride: the stride the conv2d is applied
      pad: how much padding the conv2d layer uses
      kern_size: the size of the conv2d kernel
    """
    super(PreActModule, self).__init__()

    batchnorm = nn.BatchNorm2d(num_features=in_chan)
    ReLU = nn.ReLU()
    Conv = nn.Conv2d(in_channels=in_chan, 
                      out_channels=out_chan, 
                      kernel_size=kern_size, 
                      stride=stride,
                      padding= pad
                      )
    self.layers = torch.nn.Sequential(batchnorm, ReLU, Conv)

  def forward(self, x):
    """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several torch nn layers and added to the original input.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
    """
    return x + self.layers(x)


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """

        super(ConvNet, self).__init__()

        kernel1 = (1,1)
        kernel3 = (3,3)

        conv0 = nn.Conv2d(in_channels=n_channels, out_channels=64, stride=1, padding=1, kernel_size=kernel3)
        PreAct1 = PreActModule(in_chan=64, out_chan=64, stride=1, pad=1, kern_size=kernel3)

        conv1 = nn.Conv2d(in_channels=64, out_channels=128, stride=1, padding=0, kernel_size=kernel1)
        maxpool1 = nn.MaxPool2d(kernel_size=kernel3,stride=2,padding=1)

        PreAct2a = PreActModule(in_chan=128, out_chan=128, stride=1, pad=1, kern_size=kernel3)
        PreAct2b = PreActModule(in_chan=128, out_chan=128, stride=1, pad=1, kern_size=kernel3)

        conv2 = nn.Conv2d(in_channels=128, out_channels=256, stride=1, padding=0, kernel_size=kernel1)
        maxpool2 = nn.MaxPool2d(kernel_size=kernel3,stride=2,padding=1)


        PreAct3a = PreActModule(in_chan=256, out_chan=256, stride=1, pad=1, kern_size=kernel3)
        PreAct3b = PreActModule(in_chan=256, out_chan=256, stride=1, pad=1, kern_size=kernel3)

        conv3 = nn.Conv2d(in_channels=256, out_channels=512, stride=1, padding=0, kernel_size=kernel1)
        maxpool3 = nn.MaxPool2d(kernel_size=kernel3,stride=2,padding=1)

        PreAct4a = PreActModule(in_chan=512, out_chan=512, stride=1, pad=1, kern_size=kernel3)
        PreAct4b = PreActModule(in_chan=512, out_chan=512, stride=1, pad=1, kern_size=kernel3)

        maxpool4 = nn.MaxPool2d(kernel_size=kernel3,stride=2,padding=1)
        
        PreAct5a = PreActModule(in_chan=512, out_chan=512, stride=1, pad=1, kern_size=kernel3)
        PreAct5b = PreActModule(in_chan=512, out_chan=512, stride=1, pad=1, kern_size=kernel3)

        maxpool5 = nn.MaxPool2d(kernel_size=kernel3,stride=2,padding=1)

        finalBatchNorm = nn.BatchNorm2d(512)
        finalRelu      = nn.ReLU()
        self.layers = nn.Sequential(
                                      conv0, 
                                      PreAct1, conv1, maxpool1,
                                      PreAct2a, PreAct2b, conv2, maxpool2,
                                      PreAct3a, PreAct3b, conv3, maxpool3,
                                      PreAct4a, PreAct4b, maxpool4,
                                      PreAct5a, PreAct5b, maxpool5,
                                      finalBatchNorm, finalRelu,
                                      nn.Flatten()
                                    )

        self.fc = nn.Linear(512,n_classes)


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
        x = self.layers(x)
        out = self.fc(x)
        
        return out
