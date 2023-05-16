#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#gpu_options = tf.GPUOptions(allow_growth = True)
#sess = tf.compat.v1.Session(config = tf.ConfigProto(gpu_options = gpu_options))
#import theano 
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'
import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D,MaxPool2D
from keras.layers import BatchNormalization, GlobalAveragePooling1D
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras import metrics
from keras.callbacks import ReduceLROnPlateau
from tcn import TCN
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

GloUse = {}
GloUse['train_file'] = ['train_FD00' + str(i) for i in range(1, 5)]
GloUse['test_file'] = ['test_FD00' + str(i) for i in range(1, 5)]
GloUse['SL'] = [15, 15, 16, 15]  # [15, 21, 16, 21]
GloUse['train_units'] = [100, 260, 100, 249]
GloUse['test_units'] = [100, 259, 100, 248]



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
    print(df)
    train_input = df.iloc[:, :-2].values.reshape(-1, 30, GloUse['SL'][n - 1])
    train_output = df.iloc[:, -1].values.reshape(-1, )
#    print(train_input)
#    print(train_output)
    
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
    print(test_output)
    
    #model.summary()
    i = Input(shape=(30, GloUse['SL'][n-1]))
    m = TCN()(i)
    m = Dense(1, activation='linear')(m)

    model = Model(inputs=[i], outputs=[m])

    model.summary()

    
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#    model.compile(loss='mean_squared_error', optimizer=adam)
    model.compile('adam', 'mse')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode='auto')
    history1 = model.fit(train_input, train_output, batch_size=512, epochs=100, shuffle=True)
    
    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile('adam', 'mse')
#    model.compile(loss='mean_squared_error', optimizer=adam)
    history2 = model.fit(train_input, train_output, validation_split=0.33, batch_size=512, epochs=25, shuffle=True)

    
    
    np.savetxt(GloUse['train_file'][n - 1] + "iteration.txt", history1.history['loss'] + history2.history['loss'])
    
    plt.plot((history1.history['loss']) + (history2.history['loss']), label='TCN train 0~200')
    plt.legend(loc='upper right')
    plt.show()
    
    model.save(GloUse['train_file'][n - 1] + 'TCNmodel.h5')
    
    test_predict = model.predict(test_input)
    np.savetxt(GloUse['train_file'][n - 1] + "prediction result_tcn.txt", test_predict)
    RMSE = math.sqrt(mean_squared_error(test_output, test_predict))
    SCORE = score(test_output, test_predict)
    print("test rmse:", RMSE)
    print("test score:", SCORE)

    np.savetxt(GloUse['train_file'][n - 1] + "RMSE_tcn.txt", [RMSE])
    np.savetxt(GloUse['train_file'][n - 1] + "SCORE_tcn.txt", [SCORE])




