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

from recognizer import *

if __name__ == '__main__':
    data = pd.read_csv("Data/data.csv")
    train_data = data.sample(frac=0.98, random_state=200)
    test_data = data.drop(train_data.index)
    train_labels = nd.array(train_data.iloc[:,0])
    train_features = nd.array(train_data.iloc[:,1:785]) 
    test_labels = nd.array(test_data.iloc[:,0])
    test_features = nd.array(test_data.iloc[:,1:785])

    print("Train - 1, Test - 2")
    mode = input()

    if int(mode) == 1:  
        deep_net = Recognizer()

        dataset = gluon.data.ArrayDataset(train_features,train_labels)
        train_iter = gluon.data.DataLoader(dataset,256,shuffle=True)
        
        loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(deep_net.net.collect_params(),'sgd',{'learning_rate' : 0.1} )

        deep_net.train(10, #num_epchos
                        train_iter,
                        loss_function,
                        trainer,
                        256) #batch_size

        deep_net.save_parameters('recognizer.params')

    elif int(mode) == 2:
        deep_net = Recognizer()
        deep_net.load_parameters('recognizer.params')

    
    accuracy = 0
    for i in range(test_labels.shape[0]):        
        if test_labels[i].asnumpy()[0] == deep_net.net(test_features)[i].asnumpy().argmax():
            accuracy += 1

    print("Net accuracy: ",100*(accuracy/test_labels.shape[0]),"%")