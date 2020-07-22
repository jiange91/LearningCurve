import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

'''
Dataset: MNIST
Time: 68s
Acuracy: 90%
'''


def loadData(fileName):
    print('start reading file')

    data = np.genfromtxt(fileName, delimiter=',')
    Label = data[:, 0]
    X = data[:, 1:] / 255

    return X, Label


def labelVotes(arr):
    return np.bincount(arr).argmax()

def model_validation_kdTree(trainDataArr, trainLabelArr, testDataArr, testLabelArr, topK):
    print('start testing')
    
    kdt = KDTree(trainDataArr, leaf_size=30, metric='euclidean')
    idx = kdt.query(testDataArr, k=topK, return_distance=False)

    y = trainLabelArr[np.apply_along_axis(labelVotes, 1, idx)]
    return 1 - (np.sum(y!=testLabelArr) / len(testLabelArr))


if __name__ == '__main__':
    start = time.time()

    # 获取训练集
    trainDataArr, trainLabelArr = loadData('../../../dataset/mnist_train.csv')
    # 获取测试集
    testDataArr, testLabelArr = loadData('../../../dataset/mnist_test.csv')

    # trainDataArr, trainLabelArr, testDataArr, testLabelArr = genData(1000, 3, 2)
    # 计算测试集正确率
    accur = model_validation_kdTree(trainDataArr, trainLabelArr.T, testDataArr[:200], testLabelArr.T[:200], 25)
    # 打印正确率
    print('accur is:%d' % (accur * 100), '%')

    end = time.time()
    # 显示花费时间
    print('time span:', end - start)
