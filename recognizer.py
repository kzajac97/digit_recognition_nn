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

import random

ctx = mx.cpu()

train_data = pd.read_csv("Data/train.csv")
test_data = pd.read_csv("Data/test.csv")
train_labels = nd.array(train_data.iloc[:,0])
train_features = nd.array(train_data.iloc[:,1:785]) 

batch_size = 256
dataset = gluon.data.ArrayDataset(train_features,train_labels)
train_iter = gluon.data.DataLoader(dataset,batch_size,shuffle=True)

net = nn.Sequential()
net.add(nn.Dense(784,activation='relu'))
net.add(nn.Dense(300,activation='relu'))
net.add(nn.Dense(200,activation='relu'))
net.add(nn.Dense(100,activation='relu'))
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate' : 0.1} )

num_epochs = 10
if __name__ == '__main__':
    for epoch in range(1,num_epochs+1):
        for data,label in train_iter:
            with autograd.record():
                y_hat = net(data)
                loss_value = loss_function(y_hat,label).sum()
                
            loss_value.backward()
            trainer.step(batch_size)

    for _ in range(10):
        r = random.randint(0,40000)

        print(train_labels[r])
        print(net(train_features)[r].argmax())