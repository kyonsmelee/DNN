#-*- coding:utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers.core import Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import plot_model
import matplotlib.pyplot as plt
import math
import sys
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

if os.path.exists(MODEL_DIR) is False:
    os.mkdir(MODEL_DIR)

argvs = sys.argv
data_number = int(argvs[1])#csvファイルの1列に使うデータの個数:5400
csv_row = int(argvs[2])#csvファイルの列の数:21
input_data = int(argvs[3])#入力次元数:540
shift_number = int(argvs[4])#データのずらす数:60
shift_count = int(math.floor((data_number - input_data)/shift_number) + 1) #ずらす回数

n_hidden = 100 #隠れ層の個数
output_data = 4 #出力次元数
input_number = shift_count * csv_row #入力の回数
epoch = 500
batchSize = 100

train_number = input_number * 12 * 0.8 #学習データ数
test_number = input_number * 12  * 0.16 #テストデータ数

fname = 'data={},shift_number={},hidden={},epoch={}'.format(data_number*csv_row,shift_number,n_hidden,epoch)

list_rssi = [ ]
list_norm = [ ]

def getdata(filename):
    list_shift_data = [ ]
    list_return_data = [ ]
    list_Y = [ ]

    for i in range(csv_row): #csvファイルの列数分
        data = np.loadtxt("/Users/kyonsu/Desktop/Tensorflow/{}.csv".format(filename),delimiter=",",usecols=(i+1))
        for j in range(shift_count): #入力データのずらし
            list_shift_data.append(data[shift_number*j:shift_number*j+input_data])
            list_Y.append(CorrectData(filename))

    list_norm = (list_shift_data / np.max(list_shift_data)) #正規化
    list_return_data.append(list_norm)
    list_return_data.append(list_Y)
    return list_return_data

def CorrectData(filename): #正解データ作成関数
    data_split = filename.split('_')
    #print data_split[2]
    if int(data_split[2]) == 0:
        Y = [1,0,0,0]
    elif int(data_split[2]) == 10:
        Y = [0,1,0,0]
    elif int(data_split[2]) == 40:
        Y = [0,0,1,0]
    else:
        Y = [0,0,0,1]

    return Y

'''
Y1 = np.array([[1,0,0,0]for i in range(input_number)])
Y2 = np.array([[0,1,0,0]for i in range(input_number)])
Y3 = np.array([[0,0,1,0]for i in range(input_number)])
Y4 = np.array([[0,0,0,1]for i in range(input_number)])
'''

rssi_1765_0 = getdata("rssi_1765_0")
rssi_1765_10 = getdata("rssi_1765_10")
rssi_1765_40 = getdata("rssi_1765_40")
rssi_1765_50 = getdata("rssi_1765_50")
rssi_1770_0 = getdata("rssi_1770_0")
rssi_1770_10 = getdata("rssi_1770_10")
rssi_1770_40 = getdata("rssi_1770_40")
rssi_1770_50 = getdata("rssi_1770_50")
rssi_1775_0 = getdata("rssi_1775_0")
rssi_1775_10 = getdata("rssi_1775_10")
rssi_1775_40 = getdata("rssi_1775_40")
rssi_1775_50 = getdata("rssi_1775_50")

X = np.concatenate((rssi_1765_0[0],rssi_1765_10[0],rssi_1765_40[0],rssi_1765_50[0],
                   rssi_1770_0[0],rssi_1770_10[0],rssi_1770_40[0],rssi_1770_50[0],
                   rssi_1775_0[0],rssi_1775_10[0],rssi_1775_40[0],rssi_1775_50[0]),axis=0)

Y = np.concatenate((rssi_1765_0[1],rssi_1765_10[1],rssi_1765_40[1],rssi_1765_50[1],
                   rssi_1770_0[1],rssi_1770_10[1],rssi_1770_40[1],rssi_1770_50[1],
                   rssi_1775_0[1],rssi_1775_10[1],rssi_1775_40[1],rssi_1775_50[1]),axis=0)



#Y = np.concatenate((Y1,Y2,Y3,Y4,Y1,Y2,Y3,Y4,Y1,Y2,Y3,Y4),axis=0)

'''学習データ64%/検証データ16%/テストデータ20%'''
print "get test_data"
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=int(train_number))
X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,test_size=int(test_number))

'''学習モデルの設定'''
print "model_function"
model = Sequential()

'''入力層-隠れ層-隠れ層-隠れ層-出力層'''
model.add(Dense(n_hidden,input_dim=input_data))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(n_hidden))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(n_hidden))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(output_data))
model.add(Activation('softmax'))

print "create_model"
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.01),metrics=['accuracy'])  #学習機の作成
early_stopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1) #学習の停止
hist = model.fit(X,Y,epochs=epoch,batch_size=batchSize,validation_data=(X_validation,Y_validation),callbacks=[early_stopping]) #モデル学習
plot_model(model, to_file='model.png',show_shapes=True,show_layer_names=True)
model.save(MODEL_DIR + '/model.hdf5')  #学習モデルの保存

'''学習の可視化'''
val_acc = hist.history['val_acc']
#val_loss = hist.history['val_loss']

fig = plt.figure()
plt.plot(range(len(val_acc)),val_acc,label='acc',color='blue')
plt.xlabel('epochs')
plt.title(fname)
plt.savefig('/Users/kyonsu/Desktop/Tensorflow/{}.png'.format(fname))

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
