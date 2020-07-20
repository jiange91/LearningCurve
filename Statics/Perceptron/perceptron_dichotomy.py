# coding=utf-8
# Author: Zijian
# Modified Date:2020-7-20

'''
The code is based on the work of Dodo
Dataset：Mnist
Trainning set：60000
Testing set：10000
------------------------------
Rel：
Accuracy：81.72%
Time span：91.7s
'''

from sklearn import preprocessing
import numpy as np
import time


def loadData(path):
    '''
    load dataset
    :param path to file
    :return: a tuple containning list of features and labels
    '''

    print('-----------< start reading data >----------')
    data = np.genfromtxt(path, delimiter=',')

    # extract label, ranging from 0-9, from the first column, since we are doing binary classification,
    # reassign the labels to -1 and 1
    label = np.where(data[:, 0] > 4, 1, -1)

    # extract features and conduct basic normalization
    X = data[:, 1:] / 255
    return X, label
    # normalized_X = preprocessing.normalize(X)
    # return normalized_X, label


def perceptron(dataArr, labelArr, iter=50):
    '''
    process for trainning a perceptron model
    :param dataArr: training_X
    :param labelArr: training_label (training_Y)
    :param iter: default 50 (no further optimization from 30, you can play around if time is available)
    :return: trained vector W and b
    '''
    print('start to trans')
    # Transform X and Y into matrix form to enhance computational efficiency
    dataMat = np.mat(dataArr)
    # Use Transpose here so that
    labelMat = np.mat(labelArr).T
    # 获取数据矩阵的大小，为m*n
    m, n = np.shape(dataMat)
    # 创建初始权重w，初始值全为0。
    # np.shape(dataMat)的返回值为m，n -> np.shape(dataMat)[1])的值即为n，与
    # 样本长度保持一致
    w = np.zeros((1, n))
    # 初始化偏置b为0
    b = 0
    # 初始化步长，也就是梯度下降过程中的n，控制梯度下降速率
    h = 0.0001

    # 进行iter次迭代计算
    for k in range(iter):
        # 对于每一个样本进行梯度下降
        # 李航书中在2.3.1开头部分使用的梯度下降，是全部样本都算一遍以后，统一
        # 进行一次梯度下降
        # 在2.3.1的后半部分可以看到（例如公式2.6 2.7），求和符号没有了，此时用
        # 的是随机梯度下降，即计算一个样本就针对该样本进行一次梯度下降。
        # 两者的差异各有千秋，但较为常用的是随机梯度下降。
        for i in range(m):
            # 获取当前样本的向量
            xi = dataMat[i]
            # 获取当前样本所对应的标签
            yi = labelMat[i]
            # 判断是否是误分类样本
            # 误分类样本特诊为： -yi(w*xi+b)>=0，详细可参考书中2.2.2小节
            # 在书的公式中写的是>0，实际上如果=0，说明改点在超平面上，也是不正确的
            if -1 * yi * (w * xi.T + b) >= 0:
                # 对于误分类样本，进行梯度下降，更新w和b
                w = w + h * yi * xi
                b = b + h * yi
        # 打印训练进度
        print('Round %d:%d training' % (k, iter))

    # 返回训练完的w、b
    return w, b


def model_test(dataArr, labelArr, w, b):
    '''
    测试准确率
    :param dataArr:测试集
    :param labelArr: 测试集标签
    :param w: 训练获得的权重w
    :param b: 训练获得的偏置b
    :return: 正确率
    '''
    print('start to test')
    # 将数据集转换为矩阵形式方便运算
    dataMat = np.mat(dataArr)
    # 将label转换为矩阵并转置，详细信息参考上文perceptron中
    # 对于这部分的解说
    labelMat = np.mat(labelArr).T

    # 获取测试数据集矩阵的大小
    m, n = np.shape(dataMat)
    # 错误样本数计数
    errorCnt = 0
    # 遍历所有测试样本
    for i in range(m):
        # 获得单个样本向量
        xi = dataMat[i]
        # 获得该样本标记
        yi = labelMat[i]
        # 获得运算结果
        result = -1 * yi * (w * xi.T + b)
        # 如果-yi(w*xi+b)>=0，说明该样本被误分类，错误样本数加一
        if result >= 0: errorCnt += 1
    # 正确率 = 1 - （样本分类错误数 / 样本总数）
    accruRate = 1 - (errorCnt / m)
    # 返回正确率
    return accruRate


if __name__ == '__main__':
    # 获取当前时间
    # 在文末同样获取当前时间，两时间差即为程序运行时间
    start = time.time()

    # 获取训练集及标签
    trainData, trainLabel = loadData('../Mnist/mnist_train.csv')
    # 获取测试集及标签
    testData, testLabel = loadData('../Mnist/mnist_test.csv')

    # 训练获得权重
    w, b = perceptron(trainData, trainLabel, iter=50)
    # 进行测试，获得正确率
    accruRate = model_test(testData, testLabel, w, b)

    # 获取当前时间，作为结束时间
    end = time.time()
    # 显示正确率
    print('accuracy rate is:', accruRate)
    # 显示用时时长
    print('time span:', end - start)
