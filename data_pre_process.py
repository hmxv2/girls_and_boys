import json
import csv
import pandas as pd
import numpy as np
import sys
import pickle
import random
import math

fi=sys.argv[1]#./data
fo=sys.argv[2]#./data_pre_process

#read
train_data_=pd.read_csv(fi+'/train.csv', header=None)
test_data_=pd.read_csv(fi+'/test.csv', header=None)

train_data=train_data_
test_data=test_data_

print(train_data.shape, test_data.shape)


rows, cols=train_data.shape
cols-=2
labels=np.array(train_data.iloc[:,cols]).tolist()
data=[]    #data: all data
for col in range(cols):
    data1=np.array(train_data.iloc[:,col]).tolist()
    data2=np.array(test_data.iloc[:,col]).tolist()
    data.append(data1+data2)
all_cnt=0    #all sample property
digit_cnt=0    #digital property
digit_with_little_empty_cnt=0    #digital property with little empty blank
not_digit_cnt=0    #sample property as string
not_digit_extract_cnt=0    #string property with small capacity
padding_cnt=0    #padding
lock=0
first_col_monitor=[]


result=[]
result_not_digit=[]
for ii in range(len(data)):
    all_cnt+=1
    if not(ii%10):
        print('processed %3.2f %%' %(ii*100/len(data)))#print processed percentage
    try:
        temp=float(data[ii][0])#if data[ii][0] is string then exception occurs
        #
        digit_cnt+=1
        l=[]
        for row in range(len(data[ii])):
            try:
                tmp=float(data[ii][row])    #to be robustness
            except:
                tmp=0.0
            l.append(tmp)

        #nan
        isnan_cnt=0
        value_sum=0.0
        l_not_nan=[]
        for value in l:
            if math.isnan(value):    #find nan type and count
                isnan_cnt+=1
            else:
                l_not_nan.append(value)
                value_sum+=value
        if l_not_nan==[]:
            print('empty column!')
            continue
        max_l=max(l_not_nan)
        min_l=min(l_not_nan)
        if max_l-min_l<0.001:
            print('column %s :max is almost equal to min' %ii)
            pass
        else:
            for idx, value in enumerate(l):
                if math.isnan(value):
                    l[idx]=-1
                else:
                    l[idx]=(value-min_l)/(max_l-min_l)
            result.append(l)
            digit_with_little_empty_cnt+=1
    except:
        not_digit_cnt+=1

        data_col=data[ii]
        #statistic using a dictionary
        statistic_dict={}
        for w in data_col:
            if w in statistic_dict:
                pass
            else:
                statistic_dict[w]=len(statistic_dict)
        if len(statistic_dict)>0.01*len(data_col):    #give up
            continue
        #build inverse index
        idx_inv={}
        for idx,w in enumerate(data_col):
            wi=statistic_dict[w]
            if wi in idx_inv:
                idx_inv[wi].append(idx)
            else:
                idx_inv[wi]=[idx]
        #padding
        for idx in idx_inv:
            l=[0]*len(data_col)
            #print(l[0:20])
            for x in idx_inv[idx]:
                l[x]=1
            #print(l[0:20])
            result_not_digit.append(l)
            padding_cnt+=1
        not_digit_extract_cnt+=1
#merge
for l in result:
    result_not_digit.append(l)
result=result_not_digit

#print statistic        
print('%s columns in raw data' %all_cnt)
print('%s columns contain digit data and with blanks less than ten persentage' %digit_cnt)
print('%s columns contain string data' %not_digit_cnt)
print('remain %s columns with string data' %not_digit_extract_cnt, 'and padding them to be %s columns' %padding_cnt, len(result_not_digit))
print('finally get %s columns' %len(result))


rows, cols=train_data.shape
cols-=2
#train and test data
data_train=[]
data_test=[]
for data_col in result:
    data_train.append(data_col[0:rows])
    data_test.append(data_col[rows:])
    
#label
positive_sample_cnt=0
labels_tmp=[0]*len(labels)
for idx, label in enumerate(labels):
    if label==1:
        labels_tmp[idx]=1
        positive_sample_cnt+=1
#print
print('positive samples: %s' %positive_sample_cnt)
print('property in training sample: %s' %len(data_train), '  property in testing sample: %s' %len(data_test))
print('training samples: %s' %len(data_train[0]),
      '  training sample labels: %s' %len(labels),
      '  testing samples: %s' %len(data_test[0]))

#save
print('saving...')
with open(fo+'/train_data.js', 'wb') as f:
    pickle.dump(data_train, f)
f.close

with open(fo+'/test_data.js', 'wb') as f:
    pickle.dump(data_test, f)
f.close()

with open(fo+'/labels.js', 'wb') as f:
    pickle.dump(labels_tmp, f)
f.close()
print('saved!')