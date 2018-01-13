# 导入库
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1./(1 + np.exp(-z))

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
    print(sigmoid(-100))