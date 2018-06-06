import json
import csv
import pandas as pd
import numpy as np
import sys
import pickle
import random
import math

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score

from Net import Net

import sys
#sys.path.append('/data1/hmx/softwave/hyperboard')
#from hyperboard import Agent

use_cuda=torch.cuda.is_available()
torch.manual_seed(1)

#input data file path and models saving file path
fi=sys.argv[1]#./data_pre_process
fo=sys.argv[2]#./models

#
print('reading data from path:', fi)
f=open(fi+'/train_set.js', 'rb')#./simple_process2
train_set_=pickle.load(f)
f.close()
f=open(fi+'/valid_set.js', 'rb')
valid_set_=pickle.load(f)
f.close()
f=open(fi+'/test_set.js', 'rb')
test_set_=pickle.load(f)
f.close()
print('reading finished')


train_set = train_set_.copy()
valid_set = valid_set_.copy()
test_set = test_set_.copy()
        
#train_set = np.array(train_set+valid_set, dtype=np.float32)
train_set = np.array(train_set, dtype=np.float32)
valid_set = np.array(valid_set, dtype=np.float32)
print(train_set.shape, valid_set.shape)

#hyper para
epochs=100
lr=0.0006
in_dim= 881
hidden1_dim=500
hidden2_dim=200
criterion_weight = [1,2.1]
'''
#hyperboard
agent = Agent(port = 5100)
valid_auc_para={'name':'valid_auc', 'in_dim':in_dim, 'hidden1_dim':hidden1_dim, 'hidden2_dim':hidden2_dim, 'lr':lr}
train_loss_para={'name':'train_loss', 'in_dim':in_dim, 'hidden1_dim':hidden1_dim, 'hidden2_dim':hidden2_dim, 'lr':lr}
valid_auc = agent.register(valid_auc_para, 'y1', overwrite=True)
train_loss = agent.register(train_loss_para, 'y1', overwrite=True)
'''
#training
train_set_inputs=train_set[:,0:-1]
train_set_labels=train_set[:,-1]
print(train_set_inputs.shape , train_set_labels.shape)

#to torch dataset format
torch_dataset = Data.TensorDataset(data_tensor=torch.from_numpy(train_set_inputs), 
                                   target_tensor=torch.from_numpy(train_set_labels)
                                  )  
  
#put torch_dataset into Dataloader
loader = Data.DataLoader(  
    dataset=torch_dataset,  
    batch_size=31000, #batch_size
    shuffle=False, #shuffle or not
    num_workers=2, #multi thread
    )  

#valid set
valid_set_inputs=valid_set[:,0:-1]
valid_set_labels=valid_set[:,-1]
valid_set_inputs = Variable(torch.from_numpy(valid_set_inputs))

#init a full connect nerual network
model = Net(use_cuda=use_cuda, 
            in_dim= in_dim, 
            hidden1_dim=hidden1_dim, 
            hidden2_dim=hidden2_dim
           )
if use_cuda:
    model.cuda()
    
#using cross entropy as criterion    
criterion_weight = torch.from_numpy(np.array(criterion_weight, dtype=np.float32))
if use_cuda:
    criterion_weight = criterion_weight.cuda()
        
criterion=nn.CrossEntropyLoss(criterion_weight, size_average=True)
#adam optimizer
optimizer = optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr = lr)

#training
print('training...')
for epoch in range(epochs):
    for step, (train_set_inputs, train_set_labels) in enumerate(loader):
        train_set_inputs = Variable(train_set_inputs)
        train_set_labels = Variable(train_set_labels).long()
        if use_cuda:
            train_set_labels = train_set_labels.cuda()
        #
        predicts = model(train_set_inputs)
        loss = criterion(predicts, train_set_labels)
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #valid set
    if (epoch%2)==0:
        model.eval()
        valid_predicts = model(valid_set_inputs)
        if use_cuda:
            pred = valid_predicts.data.cpu().numpy()
            pred = pred[:,1]
        else:
            pred = valid_predicts.data.numpy()[:,1]
        
        pred = pred.tolist()
        targets =  valid_set_labels.tolist()
        auc_score=roc_auc_score(targets, pred)
        print('saving model: train_set_loss-{:.4f} valid_set_auc-{:.4f}.model'.format(loss.data[0], auc_score))
        #
        #agent.append(valid_auc, epoch, auc_score)
        #agent.append(train_loss, epoch, loss.data[0])
        #save model
        torch.save(model.state_dict(), fo+'/train_set_loss-{:.4f} valid_set_auc-{:.4f}.model'.format(loss.data[0], auc_score))
        #for training
        model.train()
print('training finished')