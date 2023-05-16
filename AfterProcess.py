import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import tcn

GloUse = {}
GloUse['train_file'] = ['train_FD00' + str(i) for i in range(1, 5)]
GloUse['test_file'] = ['test_FD00' + str(i) for i in range(1, 5)]
GloUse['SL'] = [15, 15, 16, 15]  # [15, 21, 16, 21]
GloUse['train_units'] = [100, 260, 100, 249]
GloUse['test_units'] = [100, 259, 100, 248]


# 绘制制定unit的对比图象
def DrawUnit(df, No, model):
    test_input = df[df.unit == No].iloc[:, :-2].values.reshape((-1, 30, GloUse['SL'][n - 1]))
    test_output = df[df.unit == No].iloc[:, -1].values.reshape(-1, )
    test_predict = model.predict(test_input)
    plt.figure(1, figsize=(15, 9))
    plt.title('test unit #' + str(No))
    plt.plot(test_output, 'ob', ms=3)
    plt.plot(test_output, 'b', lw=2, label='actual')
    plt.legend(loc='lower left')
    plt.plot(test_predict, 'or', ms=3)
    plt.plot(test_predict, 'r', lw=2, label='prediction')
    plt.legend(loc='lower left')
    plt.show()


if __name__ == '__main__':
    n = 1
    # 构建测试数据
    df = pd.read_pickle('Data/' + GloUse['test_file'][n - 1] + ".pickle")
    df1 = []
    df2 = []
    for i in range(GloUse['test_units'][n - 1]):
        if (i + 1) in (df.unit.values):
            df1.append(df[df.unit == i + 1].iloc[-1, :-2].values)
            df2.append(df[df.unit == i + 1].iloc[-1, -1])
    test_input = np.array(df1).reshape(-1, 30, GloUse['SL'][n - 1])
    test_output = np.array(df2).reshape(-1, )
    
    df = pd.read_pickle('Data/' + GloUse['train_file'][n - 1] + '.pickle')
    
    train_input = df.iloc[:, :-2].values.reshape(-1, 30, GloUse['SL'][n - 1])
    train_output = df.iloc[:, -1].values.reshape(-1, )
    plt.plot(df[df.unit == 1].label)
    plt.xlabel("Time Cycle")
    plt.ylabel("Remaining Useful Life")
    plt.show()
    # 加载模型
    model = load_model('train_FD001TCNmodel.h5')
    # 得出预测结果
#    test_predict = model.predict(test_input)
    test_predict = np.loadtxt(GloUse['train_file'][n - 1] + "prediction result_tcn.txt")
    # 绘制对比图象
    index = np.argsort(test_output)
    test_output = np.sort(test_output)
#    test_predict = np.sort(test_predict)
    test_predict = np.array([test_predict[i] for i in index])
                    

    plt.figure(1, figsize=(16, 8))
    plt.title('Comparison between Predicted and True RUL',fontsize = 15,fontweight = 'bold')
    plt.plot(test_output, 'bo', ms=2)
    plt.plot(test_output, 'b-', lw=1, label='Actual RUL')
    plt.legend(loc='upper right')
    plt.plot(test_predict, 'o', ms=2)
    plt.plot(test_predict, '-', lw=1, label='Predicted RUL')
    plt.legend(loc='upper right')
    plt.xlabel("Time Cycle",fontsize = 15,fontweight = 'bold')
    plt.ylabel("Remaining Useful Life",fontsize = 15,fontweight = 'bold')

    plt.show()

    DrawUnit(df, 21, model)
    DrawUnit(df, 24, model)
    DrawUnit(df, 34, model)
    DrawUnit(df, 81, model)
