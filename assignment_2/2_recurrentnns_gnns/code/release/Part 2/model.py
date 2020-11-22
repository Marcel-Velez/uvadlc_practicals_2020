# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        
        self.batch_size = batch_size
        self.seq_length = 
        self.vocabulary_size = vocabulary_size

        self.model = nn.LSTM(batch_size, lstm_num_hidden, lstm_num_layers, batch_first=True)
        self.emb = nn.Embedding(vocabulary_size, 128) # A NICE POWER OF 2 NUMBER

    def forward(self, x):
        # Implementation here...
        for t in range(self.seq_length-1):
            x, hidden = self.model(self.emb(x[:,t]))

        return x

    def generate(self, start_sentence=None, genLen=30):
        def backtranslate():
            raise NotImplementedError

        genSen = []
        if not start_sentence:
            start_sentence = torch.random((1))
        
        for t in range(genLen):
            x = self.emb(start_sentence)
            output = self.model(x)
            if t >= (len(start_sentence)-1):
                start_sentence.append(output)

        return backtranslate(genSen)

