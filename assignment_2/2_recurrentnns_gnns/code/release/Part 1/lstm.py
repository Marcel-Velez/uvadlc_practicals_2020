"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTM, self).__init__()
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = device
        self.gx = nn.Parameter(torch.normal(0, 0.0001, size=(hidden_dim, input_dim)))
        self.gh = nn.Parameter(torch.normal(0, 0.0001, size=(hidden_dim, hidden_dim)))
        self.gb = nn.Parameter(torch.normal(0, 0.0001, size=(hidden_dim, 1)))

        self.ix = nn.Parameter(torch.normal(0, 0.0001, size=self.gx.shape))
        self.ih = nn.Parameter(torch.normal(0, 0.0001, size=self.gh.shape))
        self.ib = nn.Parameter(torch.normal(0, 0.0001, size=self.gb.shape))

        self.fx = nn.Parameter(torch.normal(0, 0.0001, size=self.gx.shape))
        self.fh = nn.Parameter(torch.normal(0, 0.0001, size=self.gh.shape))
        self.fb = nn.Parameter(torch.normal(0, 0.0001, size=self.gb.shape))

        self.ox = nn.Parameter(torch.normal(0, 0.0001, size=self.gx.shape))
        self.oh = nn.Parameter(torch.normal(0, 0.0001, size=self.gh.shape))
        self.ob = nn.Parameter(torch.normal(0, 0.0001, size=self.gb.shape))

        self.ph = nn.Parameter(torch.normal(0, 0.0001, size=(num_classes,hidden_dim)).T)
        self.pb = nn.Parameter(torch.normal(0, 0.0001, size=(num_classes, 1)))

        self.emb =  nn.Embedding(num_classes, 1)
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
            # print("cur batch: {}".format(x))
            # print("cur first x: {}".format(x[:,t,0]))
            cur_x = self.emb(x[:,t].long()) if len(x.shape) == 2 else self.emb(x[:,t,:].long())# cur_x = (batch_size, num classes) want onehot encoded voor batch van 256
            print("cur_x: {}".format(cur_x))
            # print("embedded input shape: {}".format(cur_x.shape))
            # print("gx: {}".format(self.gx.shape))
            # print("gh: {}".format(self.gh.shape))
            # print("h_prev: {}".format(h_prev.shape))

            first = self.gx @ cur_x.T
            # print("first shape: {}\n".format(first.shape))

            # print("cur_x: {}".format(cur_x.T.shape))
            
            g = self.tan(self.gx @ cur_x.T + self.gh @ h_prev + self.gb)
            i = self.sig(self.ix @ cur_x.T + self.ih @ h_prev + self.ib)
            f = self.sig(self.fx @ cur_x.T + self.fh @ h_prev + self.fb)
            o = self.sig(self.ox @ cur_x.T + self.oh @ h_prev + self.ob)
            c = g*i + c_prev *f
            h = self.tan(c) * o
            h_prev = h
            c_prev = c
        
        p = self.ph @ h + self.pb
        y = self.soft(p)
        return y.T
        ########################
        # END OF YOUR CODE    #
        #######################
