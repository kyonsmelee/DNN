#-*- coding:utf-8 -*-
import math
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.layers.core import Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import load_model

argvs = sys.argv
data_number = int(argvs[1])#csvファイルの1列に使うデータの個数
csv_row = int(argvs[2])#csvファイルの列の数
input_data = int(argvs[3])#入力次元数
shift_number = int(argvs[4])#データのずらす数
shift_count = int(math.floor((data_number - input_data)/shift_number) + 1)
batchSize = 100
epoch = 500

list_rssi = [ ]
list_shift_data = [ ]
list_norm = [ ]
list_result = [ ]
list_result_0 = [ ]
list_result_10 = [ ]
list_result_40 = [ ]
list_result_50 = [ ]

def getdata(filename):
    for i in range(csv_row): #csvファイルの列数分
        data = np.loadtxt("/Users/kyonsu/Desktop/Tensorflow/{}.csv".format(filename),delimiter=",",usecols=(i+1))
        for j in range(shift_count): #入力データのずらし
            list_shift_data.append(data[shift_number*j:shift_number*j+input_data])

    list_norm = (list_shift_data / np.max(list_shift_data)) #正規化
    return list_norm


X = getdata("test2")

model = load_model('/Users/kyonsu/Desktop/Tensorflow/model_check.hdf5')
prob = model.predict_proba(X,batch_size=batchSize,verbose=1)
classes = model.predict_classes(X[0:],batch_size=batchSize,verbose=1)

for i in range(len(classes)):
    if classes[i]==0:
        list_result_0.append("0people")
        #print "0peple"
    elif classes[i]==1:
        list_result_10.append("10people")
        #print "10people"
    elif classes[i]==2:
        list_result_40.append("40people")
        #print "40people"
    else:
        list_result_50.append("50people")
        #print "50people"
list_result.append(len(list_result_0))
list_result.append(len(list_result_10))
list_result.append(len(list_result_40))
list_result.append(len(list_result_50))
print "\n"
print list_result
#print classes
#print prob
#print classes
