import json
import csv
import pandas as pd
import numpy as np
import sys
import pickle
import random
import math
import re

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as Data

class Net(nn.Module):
    def __init__(self, use_cuda, in_dim, hidden1_dim, hidden2_dim):
        super(Net,self).__init__()
        
        self.use_cuda = use_cuda
        
        self.cut_idx = 695
        
        self.layer1_1=nn.Linear(self.cut_idx, 400)
        self.layer1_2=nn.Linear(400, 200)
        self.layer1_3=nn.Linear(200,50)
        
        self.layer2_1=nn.Linear(in_dim-self.cut_idx, 180)
        self.layer2_2=nn.Linear(180, 140)
        
        self.layer3=nn.Linear(190, 150)
        
        self.layer4=nn.Linear(150, 80)
        
        self.layer5=nn.Linear(80, 2)
        
        self.activate_fun=nn.ReLU()#.Sigmoid()
        self.softmax=nn.Softmax()
        self.dropout=nn.Dropout(0.5)
        
    def forward(self,x):
        
        if self.use_cuda:
            x=x.cuda()
        
        x1=x[:,0:self.cut_idx]
        x2=x[:,self.cut_idx:]
        
        x1=self.layer1_1(x1)
        x1=self.activate_fun(x1)
        x1=self.dropout(x1)
        x1=self.layer1_2(x1)
        x1=self.activate_fun(x1)
        x1=self.dropout(x1)
        x1=self.layer1_3(x1)
        x1=self.activate_fun(x1)
        
        x2=self.layer2_1(x2)
        x2=self.activate_fun(x2)
        x2=self.dropout(x2)
        x2=self.layer2_2(x2)
        x2=self.activate_fun(x2)
        
        x=torch.cat((x1,x2), 1)
        
        x=self.layer3(x)
        x=self.dropout(x)
        x=self.layer4(x)
        x=self.layer5(x)
        x=self.softmax(x)
        
        return x
        '''
        #print ('1------------    ', x)
        #y=self.dropout(x)
        #y=self.layer1(y)
        y=self.layer1(x)
        y=self.activate_fun(y)
        #print ('2------------    ', y)
        y=self.dropout(y)
        y=self.layer2(y)
        y=self.activate_fun(y)
      
        #print ('3------------    ', y)
        y=self.dropout(y)
        y=self.layer3(y)
        
        y=self.softmax(y)
        return y
        '''