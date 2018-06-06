import json
import csv
import pandas as pd
import numpy as np
import sys
import pickle
import random
import math
import re

from sklearn.metrics import roc_auc_score
import xgboost as xgb
import time

#test set file path, model file path and test result saving path
fi_test_set=sys.argv[1]#./simple_process2
fi=sys.argv[2]#./xgb_models/valid_set_auc0.8745050638824763-is_valid1-max_depth300-scale_weight340-nestmator180-numround180.model
fo=sys.argv[3]#./test_predict_result

f=open(fi_test_set+'/test_set.js', 'rb')
test_set=pickle.load(f)
f.close()

test_set = np.array(test_set, dtype=np.float32)
dtest = xgb.DMatrix(test_set)

xgb_model = xgb.Booster({'nthread':4}) #init model
xgb_model.load_model(fi) # load data
preds = xgb_model.predict(dtest)

#save
print('saving...')
fo_path=fo+re.findall(r'valid.+model', fi)[0][0:-6]+'.csv'
with open(fo_path, 'w') as csvfile: 
    writer = csv.writer(csvfile)
    #columns name
    writer.writerow(["Id","label"])
    #data
    for idx, x in enumerate(preds):
        writer.writerow([str((idx+1)), str(x)])
csvfile.close()
print('saving finished')
