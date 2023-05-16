#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten,Input,add,Activation
from keras.layers import Conv2D,MaxPool2D
from keras.layers import Conv1D,MaxPooling1D
from keras.layers import BatchNormalization, GlobalAveragePooling1D
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras import metrics
from keras.callbacks import ReduceLROnPlateau

GloUse = {}
GloUse['train_file'] = ['train_FD00' + str(i) for i in range(1, 5)]
GloUse['test_file'] = ['test_FD00' + str(i) for i in range(1, 5)]
GloUse['SL'] = [15, 15, 16, 15]  # [15, 21, 16, 21]
GloUse['train_units'] = [100, 260, 100, 249]
GloUse['test_units'] = [100, 259, 100, 248]

def ResBlock(x,filters,kernel_size,dilation_rate):
   r=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu')(x) #第一卷积
   r=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate)(r) #第二卷积
   #r=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate)(r)
   #r=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate)(r)
   if x.shape[-1]==filters:
       shortcut=x
   else:
       shortcut=Conv1D(filters,kernel_size,padding='same')(x)  #shortcut（捷径）
   o=add([r,shortcut])
   o=Activation('relu')(o)  #激活函数
   return o


def mean_squared_error(x, y):
    sum = 0
    n = len(x)
    for i, j in zip(x, y):
        sum = sum + (i - j) ** 2
    return sum / n


def score(x, y):
    sum = 0
    for i, j in zip(x, y):
        z = i - j
        if z < 0:
            sum = sum + np.e ** (-z / 13) - 1
        else:
            sum = sum + np.e ** (z / 10) - 1
    return sum



if __name__ == '__main__':
    n = 1
    # 构建训练数据
    df = pd.read_pickle('Data/' + GloUse['train_file'][n - 1] + '.pickle')
    
    train_input = df.iloc[:, :-2].values.reshape(-1, 30, GloUse['SL'][n - 1])
    train_output = df.iloc[:, -1].values.reshape(-1, )
#    print(train_input)
#    print(train_output)
    # 构建测试数据
    df = pd.read_pickle('Data/' + GloUse['test_file'][n - 1] + '.pickle')
    df1 = []
    df2 = []
    for i in range(GloUse['test_units'][n - 1]):
        if (i + 1) in (df.unit.values):
            df1.append(df[df.unit == i + 1].iloc[-1, :-2].values)
            df2.append(df[df.unit == i + 1].iloc[-1, -1])
    test_input = np.array(df1).reshape(-1, 30, GloUse['SL'][n - 1])
    test_output = np.array(df2).reshape(-1, )
#    print(test_input)
    
    # 构建模型
    input = Input(shape=(30,GloUse['SL'][n - 1]))
    x=ResBlock(input,filters=300,kernel_size=3,dilation_rate=1)
    x=ResBlock(x,filters=100,kernel_size=3,dilation_rate=2)
    x=ResBlock(x,filters=100,kernel_size=3,dilation_rate=4)
    x=Flatten()(x)
    #x=Dense(500,activation='tanh')(x)
    x=Dense(100,activation='tanh')(x)
    x=Dense(1,activation='sigmoid')(x)
    model=Model(input=input,output=x)
    #查看网络结构
    model.summary()

    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer=adam)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, mode='auto')
    history1 = model.fit(train_input, train_output, batch_size=512, epochs=200, callbacks = [reduce_lr], shuffle=True)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer=adam)
    history2 = model.fit(train_input, train_output, validation_split=0.33, batch_size=512, epochs=50, shuffle=True)
#    test_loss, accuracy = model.evaluate(test_input, test_output)
#    print('test loss: ', test_loss)
    
    # 保存迭代损失值
    np.savetxt(GloUse['train_file'][n - 1] + "iteration.txt", history1.history['loss'] + history2.history['loss'])
    # 绘制收敛曲线图
    plt.plot((history1.history['loss']) + (history2.history['loss']), label='DCNN train 0~250')
    plt.legend(loc='upper right')
    plt.show()
    # 保存模型
    model.save(GloUse['train_file'][n - 1] + '1dDCNNmodel.h5')
    # 保存预测值，RMSE，得分
    test_predict = model.predict(test_input)
    np.savetxt(GloUse['train_file'][n - 1] + "prediction result.txt", test_predict)
    RMSE = math.sqrt(mean_squared_error(test_output, test_predict))
    SCORE = score(test_output, test_predict)
    print("test rmse:", RMSE)
    print("test score:", SCORE)

    np.savetxt(GloUse['train_file'][n - 1] + "RMSE.txt", [RMSE])
    np.savetxt(GloUse['train_file'][n - 1] + "SCORE.txt", [SCORE])




