import json
import csv
import pandas as pd
import numpy as np
import sys
import pickle
import random
import math

from sklearn.metrics import roc_auc_score
import xgboost as xgb
import time

#input data file path and models saving file path
fi=sys.argv[1]#./simple_process2
fo=sys.argv[2]#./models
is_valid = sys.argv[3]#is_valid = 0, means training model using train set and valid set together

#
print('reading data from path:', fi)
f=open(fi+'/train_set.js', 'rb')#./simple_process2
train_set_=pickle.load(f)
f.close()
f=open(fi+'/valid_set.js', 'rb')
valid_set_=pickle.load(f)
f.close()
print('reading finished')

train_set = train_set_.copy()
valid_set = valid_set_.copy()

#train_set = np.array(train_set+valid_set, dtype=np.float32)
train_set = np.array(train_set, dtype=np.float32)
valid_set = np.array(valid_set, dtype=np.float32)
print('train set shape: ', train_set.shape, '  valid set shape: ', valid_set.shape)


start_time = time.time()
#read in data
if is_valid==0:
    train_set = np.concatenate((train_set, valid_set), axis = 0)
    
train_set_label = train_set[:,-1]
cols =train_set_label.shape
cols=cols[0]
label = []
for x in range(cols):
    label.append(train_set_label[x])
    
dtrain = xgb.DMatrix(train_set[:,0:-1], label=label)
dvalid = xgb.DMatrix(valid_set[:,0:-1])# specify parameters via map

#hyper para
max_depth=200
scale_weight=2.2#340
n_estimator=380
num_round = 80

param = {'max_depth':max_depth, 'eta':1, 'silent':0, 'objective':'binary:logistic',
         'scale_pos_weight':scale_weight,
         'eval_metric':'auc',
        #'learning_rate':0.1,
        'min_child_weight':1.2,
        'max_delta_step':1,
        'reg_lambda':100,
        'n_estimators':n_estimator
        }
print('training...')
bst = xgb.train(param, dtrain, num_round)
print('training finished!')

# make prediction
preds = bst.predict(dvalid)
targets = valid_set[:,-1]
auc_score=roc_auc_score(targets, preds)
print('valid set AUC: ', auc_score)
model_name = 'valid_set_auc{}-is_valid{}-max_depth{}-scale_weight{}-nestmator{}-numround{}.model'.format(auc_score,
                                                                                                             is_valid,
                                                                                                             max_depth,
                                                                                                                 scale_weight,
                                                                                                                 n_estimator,
                                                                                                                 num_round
                                                                                                                )
bst.save_model(fo+'/'+model_name)#model saving
print('model saved with name: ', model_name)
print('running time: %2.5f mins' %((time.time()-start_time)/60))