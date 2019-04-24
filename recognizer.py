import sys
sys.path.insert(0,'..')

import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import init
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon import data as gluon_data
from mxnet.gluon import loss as gluon_loss

# Class representig digit recognizer
# Uses mxnet neural network library
class Recognizer(nn.Block):
    def __init__(self,**kwargs):
        super(Recognizer,self).__init__(**kwargs)
        self.ctx = mx.cpu() 

        # Four Densly connected layers with dropout(P=20%)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(784,activation='relu'))
        self.net.add(nn.Dropout(0.2))
        self.net.add(nn.Dense(392,activation='relu'))
        self.net.add(nn.Dropout(0.2))
        self.net.add(nn.Dense(196,activation='relu'))
        self.net.add(nn.Dropout(0.2))
        self.net.add(nn.Dense(98,activation='relu'))
        self.net.add(nn.Dropout(0.2))
        self.net.add(nn.Dense(10))
        self.net.initialize(init.Normal(sigma=0.01))

    # Overwrite forward pass
    def forward(self, x):
        return self.net(x)

    # train model
    def train(self,num_epochs,train_iter,loss_function,trainer,batch_size):
        for epoch in range(1,num_epochs+1):
            for data,label in train_iter:
                # autograd.record() switches dropout layers on
                with autograd.record():
                    y_hat = self.net(data)
                    loss_value = loss_function(y_hat,label).sum()
                
                loss_value.backward()
                trainer.step(batch_size)