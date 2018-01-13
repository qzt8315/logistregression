# 导入库
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 计算拟合参数的迭代次数
maxCycle = 2000
# 计算步长
alpha = 0.001

def sigmoid(z):
    ret = []
    for i in z:
        t = 1./(1 + math.exp(-i))
        ret.append(t)
    return np.matrix(ret)

# 程序开始
if __name__ == "__main__":
    # 从文本文件中读取数据
    f = open('data/data.txt')
    data = []
    # 按行读取数据
    line = f.readline()
    while line:
        data = np.append(data,[float(i) for i in line.split()])
        line = f.readline()
    # 调整矩阵的维数
    data = data.reshape((17,4))
    # 测试调整的结果
    # print(data)
    # 显示整个数据集
    # X = data[:, 1:3]
    # y = data[:, 3]
    # plt.title('watermelon data')
    # plt.xlabel('density')
    # plt.ylabel('ratio_sugar')
    # plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=100, label='bad')
    # plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=100, label='good')
    # plt.legend(loc='upper right')
    # plt.show()
    # 测试sigmoid函数
    # print(sigmoid(-100))
    # 划分测试集以及训练集
    testdata = data[0::2, 1:3]  # size:9*2
    testLabel = data[0::2, 3]  # size:9*1
    traindata = data[1::2, 1:3] # size:8*2
    trainLabel = data[1::2, 3]  # size:8*1
    # 迭代计算拟合参数
    weight = np.mat(np.ones(shape=(2,1)))
    dataMat = np.mat(traindata)
    labelMat = np.mat(trainLabel).T
    # 梯度下降求解
    for i in range(maxCycle):
        t = np.dot(dataMat, weight)
        h = sigmoid(t)
        err = labelMat - h.T  # size:8*1
        a = alpha * np.dot(dataMat.T, err)
        weight = weight + a
    # 显示计算结果
    print(weight)