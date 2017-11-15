#-*- coding:utf-8 -*-
#DNN_3career_threshold.pyで作成したモデルを未知データに入れて推定人数を出力

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
from colorama import Fore,Back,Style

argvs = sys.argv
output_timing = int(argvs[1])  #推定人数を出力するタイミング(何分かに1回)
input_data = int(argvs[2])     #入力次元数
shift_number = int(argvs[3])   #データのずらす数
csv_row = int(argvs[4])        #csvファイルの列の数
std_multiple = float(argvs[5]) #標準偏差σの倍数

batchSize = 100
epoch = 500

list_result = [ ]
list_result_10 = [ ]
list_result_15 = [ ]
list_result_50 = [ ]
list_result_100 = [ ]
list_result_110 = [ ]
list_result_120 = [ ]
list_norm = [ ]
list_shift_data = [ ]

def getdata(filename): #データ取得関数
    data = [ ]
    list_data = [ ]
    list_threshold = [ ]
    list_return_data = [ ]
    list_threshold_norm = [ ]

    for i in range(csv_row): #csvファイルの列数分
        csv_data = np.loadtxt("/Users/kyonsu/Desktop/研究/Tensorflow/{}.csv".format(filename),delimiter=",",usecols=(i+1))
        output_count = int(math.floor(int(len(csv_data))/output_timing)) #出力回数:全体データ数を推定人数を出力するデータ数で割る
        for j in range(output_count):
            for k in range(output_timing): #取得するデータ数分での平均値・標準偏差を算出するために使用するデータ数取得する
                data.append(csv_data[k+j*output_timing])
            data_ave = np.average(data)
            data_std = np.std(data)
            data_threshold = data_ave + data_std * std_multiple #閾値：平均値＋標準偏差*n

            for l in range(output_timing):
                if data[l] > data_threshold:
                    list_threshold.append(data[l])

            list_threshold_norm = list_threshold/max(list_threshold) #正規化
            shift_count = int(math.floor((len(list_threshold_norm) - input_data)/shift_number) + 1)

            for m in range(shift_count):
                list_data.append(list_threshold_norm[m*shift_number:m*shift_number+input_data])
            print len(list_threshold),shift_count
            list_threshold = [ ] #閾値超えデータの初期化
            data = [ ]

    print len(list_data)
    list_return_data.append(list_data)
    return list_return_data

data = getdata("rssi_au_10")
X = np.concatenate((data),axis=0) #これがない場合、配列を1つにまとめれないのでエラー出る

model = load_model('/Users/kyonsu/Desktop/研究/Tensorflow/model/model_3ca_threshold.hdf5')
prob = model.predict_proba(X,batch_size=batchSize,verbose=1)
classes = model.predict_classes(X,batch_size=batchSize,verbose=1)

'''人数推定方法'''
for i in range(len(classes)):
    if classes[i]==0:
        list_result_10.append("10people")
    elif classes[i]==1:
        list_result_15.append("15people")
    elif classes[i]==2:
        list_result_50.append("50people")
    elif classes[i]==3:
        list_result_100.append("100people")
    elif classes[i]==4:
        list_result_110.append("110people")
    else:
        list_result_120.append("120people")

list_result.append(len(list_result_10))
list_result.append(len(list_result_15))
list_result.append(len(list_result_50))
list_result.append(len(list_result_110))
list_result.append(len(list_result_100))
list_result.append(len(list_result_120))
print list_result
print classes

'''推定人数の出力'''
if np.argmax(list_result)==0:
    print Fore.GREEN+"10 "+Fore.WHITE+"people now!!"
elif np.argmax(list_result)==1:
    print Fore.GREEN+"15 "+Fore.WHITE+"people now!!"
elif np.argmax(list_result)==2:
    print Fore.GREEN+"50 "+Fore.WHITE+"people now!!"
elif np.argmax(list_result)==3:
    print Fore.GREEN+"100 "+Fore.WHITE+"people now!!"
elif np.argmax(list_result)==4:
    print Fore.GREEN+"110 "+Fore.WHITE+"people now!!"
else:
    print Fore.GREEN+"120 "+Fore.WHITE+"people now!!"
