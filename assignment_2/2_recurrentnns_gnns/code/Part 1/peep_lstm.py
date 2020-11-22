"""
This module implements a LSTM with peephole connections in PyTorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class peepLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(peepLSTM, self).__init__()

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = device
        self.gx = nn.Parameter(torch.normal(0, 0.0001, size=(hidden_dim, num_classes)))
        self.gh = nn.Parameter(torch.normal(0, 0.0001, size=(hidden_dim, hidden_dim)))
        self.gb = nn.Parameter(torch.normal(0, 0.0001, size=(hidden_dim, 1)))

        self.cx = nn.Parameter(torch.normal(0, 0.0001, size=(hidden_dim, num_classes)))
        self.cb = nn.Parameter(torch.normal(0, 0.0001, size=(hidden_dim, 1)))

        self.ix = nn.Parameter(torch.normal(0, 0.0001, size=self.gx.shape))
        self.ih = nn.Parameter(torch.normal(0, 0.0001, size=self.gh.shape))
        self.ib = nn.Parameter(torch.normal(0, 0.0001, size=self.gb.shape))

        self.fx = nn.Parameter(torch.normal(0, 0.0001, size=self.gx.shape))
        self.fh = nn.Parameter(torch.normal(0, 0.0001, size=self.gh.shape))
        self.fb = nn.Parameter(torch.normal(0, 0.0001, size=self.gb.shape))

        self.ox = nn.Parameter(torch.normal(0, 0.0001, size=self.gx.shape))
        self.oh = nn.Parameter(torch.normal(0, 0.0001, size=self.gh.shape))
        self.ob = nn.Parameter(torch.normal(0, 0.0001, size=self.gb.shape))

        self.ph = nn.Parameter(torch.normal(0, 0.0001, size=self.gx.shape).T)
        self.pb = nn.Parameter(torch.normal(0, 0.0001, size=(num_classes, 1)))

        self.emb =  nn.Embedding(num_classes, num_classes)
        self.sig = nn.Sigmoid()
        self.tan = nn.Tanh()
        self.soft = nn.Softmax( dim=1)

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        h_prev = torch.zeros(self.gh.shape).to(self.device)
        c_prev = torch.zeros(self.gh.shape).to(self.device)

        for t in range(self.seq_length-1):
            cur_x = self.emb(x[:,t].long()) # cur_x = (batch_size, num classes) want onehot encoded voor batch van 256

            f = self.sig(self.fx @ cur_x.T + self.fh @ c_prev + self.fb)
            i = self.sig(self.ix @ cur_x.T + self.ih @ c_prev + self.ib)
            o = self.sig(self.ox @ cur_x.T + self.oh @ c_prev + self.ob)


            # g = self.tan(self.gx @ cur_x.T + self.gh @ h_prev + self.gb)
            
            c = self.sig(self.cx @cur_x.T + self.cb) * i + c_prev *f
            h = self.tan(c) * o


            h_prev = h
            c_prev = c
        
        p = self.ph @ h + self.pb
        y = self.soft(p)
        return y.T
        ########################
        # END OF YOUR CODE    #
        #######################

