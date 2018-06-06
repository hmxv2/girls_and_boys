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
from sklearn.metrics import roc_auc_score

from Net import Net

import sys
#sys.path.append('/data1/hmx/softwave/hyperboard')
#from hyperboard import Agent

use_cuda=torch.cuda.is_available()
torch.manual_seed(1)

#test set file path, model file path and test result saving path
fi_test_set=sys.argv[1]#./simple_process2
fi=sys.argv[2]#./models/train_set_loss-0.3448 valid_set_auc-0.8686.model
fo=sys.argv[3]#./test_predict_result/

f=open(fi_test_set+'/test_set.js', 'rb')#./simple_process2
test_set=pickle.load(f)
f.close()

#hyper para
in_dim= 881
hidden1_dim=500
hidden2_dim=200

model_ = Net(use_cuda=use_cuda, 
            in_dim= in_dim, 
            hidden1_dim=hidden1_dim, 
            hidden2_dim=hidden2_dim
           )
model_trained = torch.load(fi)
model_.load_state_dict(model_trained)#load model trained 
if use_cuda:
    model_.cuda()
    
#test set
test_set_inputs = Variable(torch.Tensor(test_set))

predicts = model_(test_set_inputs)
if use_cuda:
    pred = predicts.data.cpu().numpy()#gpu to cpu firstly
    pred = pred[:,1]
else:
    pred = predicts.data.numpy()[:,1]
pred = pred.tolist()

print('saving result as .csv file')
fo_path=fo+re.findall(r'train.+model', fi)[0][0:-6]+'.csv'
with open(fo_path,'w') as csvfile:
    writer = csv.writer(csvfile)
    #columns name
    writer.writerow(["Id","label"])
    #data
    for idx, x in enumerate(pred):
        writer.writerow([str((idx+1)), str(x)])
print('result saved')
