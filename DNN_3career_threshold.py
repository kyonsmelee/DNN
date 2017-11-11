#-*- coding:utf-8 -*-
#Docomo,Softbank,auの周波数帯で計測したデータを1つのモデルとして学習させる
#学習させるデータに閾値を設定しデータを除去しモデルを作成する

import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers.core import Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import plot_model
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
import math
import sys
import os
import datetime

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

if os.path.exists(MODEL_DIR) is False:
    os.mkdir(MODEL_DIR)

argvs = sys.argv
input_data = int(argvs[1])     #入力次元数:60
shift_number = int(argvs[2])   #データのずらす数:30
csv_row = int(argvs[3])        #csvファイルの列の数:21
std_multiple = float(argvs[4]) #標準偏差σの倍数

n_hidden = 100 #隠れ層のニューロン数
output_data = 6 #出力次元数
epoch = 500
batchSize = 100
file_counter = 0

list_rssi = [ ]
list_norm = [ ]
list_cut_number = [ ]
list_small_set_length = [ ]

now = datetime.datetime.now()
time = datetime.time(now.hour,now.minute,now.second)
fname = 'threshold_multiple{}_{}.{}.{}'.format(std_multiple,now.hour,now.minute,now.second)

def getdata(filename): #データ取得関数
    list_Y = [ ]
    list_data = [ ]
    list_threshold = [ ]
    list_return_data = [ ]
    list_threshold_norm = [ ]

    for i in range(csv_row): #csvファイルの列数分
        data = np.loadtxt("/Users/kyonsu/Desktop/研究/Tensorflow/{}.csv".format(filename),delimiter=",",usecols=(i+1))
        data_ave = np.average(data)
        data_std = np.std(data)
        data_threshold = data_ave + data_std * std_multiple #閾値：平均値＋標準偏差*n

        for j in range(len(data)):
            if data[j] > data_threshold:
                list_threshold.append(data[j])

        list_threshold_norm = list_threshold/max(list_threshold) #正規化
        shift_count = int(math.floor((len(list_threshold_norm) - input_data)/shift_number) + 1)
        for k in range(shift_count):
            list_data.append(list_threshold_norm[k*shift_number:k*shift_number+input_data])
            list_Y.append(CorrectData(filename))
            
        list_threshold = [ ] #閾値超えデータの初期化
    print len(list_data)
    list_return_data.append(list_data)
    list_return_data.append(list_Y)
    return list_return_data

def CorrectData(filename): #正解データ作成関数
    data_split = filename.split('_')
    #print data_split[2]
    if int(data_split[2]) == 10:
        Y = [1,0,0,0,0,0]
    elif int(data_split[2]) == 15:
        Y = [0,1,0,0,0,0]
    elif int(data_split[2]) == 50:
        Y = [0,0,1,0,0,0]
    elif int(data_split[2]) == 100:
        Y = [0,0,0,1,0,0]
    elif int(data_split[2]) == 110:
        Y = [0,0,0,0,1,0]
    else:
        Y = [0,0,0,0,0,1]
    return Y

rssi_au_10 = getdata("rssi_au_10")
rssi_au_15 = getdata("rssi_au_15")
rssi_au_15_2 = getdata("rssi_au_15_2")
rssi_au_50 = getdata("rssi_au_50")
rssi_au_100 = getdata("rssi_au_100")
rssi_au_110 = getdata("rssi_au_110")
rssi_au_120 = getdata("rssi_au_120")
rssi_softbank_10 = getdata("rssi_softbank_10")
rssi_softbank_15 = getdata("rssi_softbank_15")
rssi_softbank_15_2 = getdata("rssi_softbank_15_2")
rssi_softbank_50 = getdata("rssi_softbank_50")
rssi_softbank_100 = getdata("rssi_softbank_100")
rssi_softbank_110 = getdata("rssi_softbank_110")
rssi_softbank_120 = getdata("rssi_softbank_120")

X = np.concatenate((rssi_au_10[0],rssi_au_15[0],rssi_au_15_2[0],rssi_au_50[0],rssi_au_100[0],rssi_au_110[0],rssi_au_120[0],rssi_softbank_10[0],rssi_softbank_15[0],rssi_softbank_15_2[0],rssi_softbank_50[0],rssi_softbank_100[0],rssi_softbank_110[0],rssi_softbank_120[0]),axis=0)
Y = np.concatenate((rssi_au_10[1],rssi_au_15[1],rssi_au_15_2[1],rssi_au_50[1],rssi_au_100[1],rssi_au_110[1],rssi_au_120[1],rssi_softbank_10[1],rssi_softbank_15[1],rssi_softbank_15_2[1],rssi_softbank_50[1],rssi_softbank_100[1],rssi_softbank_110[1],rssi_softbank_120[1]),axis=0)
print len(X),len(Y)

'''学習データ64%/検証データ16%/テストデータ20%'''
print "get test_data"
train_number = len(X) * 0.8 #学習データ数=1つのファイルのデータ数*ファイルの数(6000個のデータ)=list_cut_numberの合計*0.8
test_number = len(X) * 0.16 #テストデータ数=1つのファイルのデータ数*フファイルの数(6000個のデータ)=list_cut_numberの合計*0.16
print test_number
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=int(train_number))
X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,test_size=int(test_number))

'''学習モデルの設定'''
print "model_function"
model = Sequential()

'''入力層-隠れ層-隠れ層-隠れ層-出力層'''
model.add(Dense(n_hidden,input_dim=input_data))
#model.add(PReLU())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(n_hidden))
#model.add(PReLU())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(n_hidden))
#model.add(PReLU())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(output_data))
model.add(Activation('softmax'))

print "create_model"
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.01),metrics=['accuracy'])  #学習機の作成
early_stopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1) #学習の停止
hist = model.fit(X,Y,epochs=epoch,batch_size=batchSize,validation_data=(X_validation,Y_validation),callbacks=[early_stopping]) #モデル学習
plot_model(model, to_file='model.png',show_shapes=True,show_layer_names=True)
model.save(MODEL_DIR + '/model_3ca_threshold.hdf5')  #学習モデルの保存

'''学習の可視化'''
val_acc = hist.history['val_acc']
#val_loss = hist.history['val_loss']

fig = plt.figure()
plt.plot(range(len(val_acc)),val_acc,label='acc',color='blue')
plt.xlabel('epochs')
plt.title(fname)
plt.savefig('{}.png'.format(fname))

'''テストデータのシャッフル'''
X_,Y_ = shuffle(X_test,Y_test)

'''予測の精度'''
prob = model.predict_proba(X_[0:10],batch_size=batchSize)
print 'output probability :',
print '\n'
print prob
print Y_[0:10]
loss_and_metrics = model.evaluate(X_, Y_)
print '\n'
print(loss_and_metrics)
