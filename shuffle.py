import json
import csv
import pandas as pd
import numpy as np
import sys
import pickle
import random
import math

#file in and file out
fi=sys.argv[1]#./data_pre_process
fo=sys.argv[2]#./data_pre_process

print('reading data from path: %s' %fi)
f=open(fi+'/train_data.js', 'rb')#./simple_process2
train_data_=pickle.load(f)
f.close()
f=open(fi+'/labels.js', 'rb')
labels_=pickle.load(f)
f.close()
f=open(fi+'/test_data.js', 'rb')
test_data_=pickle.load(f)
f.close()
print('reading finished!')


print('training samples: %s' %len(train_data_[0]))
print('testing samples: %s' %len(test_data_[0]))
test_data = test_data_.copy()
all_train_data = train_data_.copy()
all_train_data.append(labels_.copy())

test_data=np.array(test_data).transpose()
all_train_data=np.array(all_train_data).transpose()

test_data=test_data[:,:-1]
all_train_data=np.delete(all_train_data, 881, axis=1)

print('testing data shape: ', test_data.shape)
print('training data shape: ', all_train_data.shape)

all_train_num, _ = all_train_data.shape
valid_set_rate = 0.2
valid_set_num = int(valid_set_rate*all_train_num)
print('%s samples will be segmented from training data to be a valid set.' %valid_set_num)

test_data=test_data.tolist()
all_train_data=all_train_data.tolist()

random.shuffle(all_train_data)
valid_set = all_train_data[0:valid_set_num]
train_set = all_train_data[valid_set_num:]


#positive samples rate in valid set and train set
print('positive samples rate in valid set: ', len(valid_set)/sum(np.array(valid_set)[:,-1]))
print('positive samples rate in train set: ', len(train_set)/sum(np.array(train_set)[:,-1]))

#save
print('saving training set, valid set, testing set data in path: ', fo)
with open(fo+'/train_set.js', 'wb') as f:
    pickle.dump(train_set, f)#./simple_process2
f.close

with open(fo+'/valid_set.js', 'wb') as f:
    pickle.dump(valid_set, f)
f.close()

with open(fo+'/test_set.js', 'wb') as f:
    pickle.dump(test_data, f)
f.close()
print('saving finished!')