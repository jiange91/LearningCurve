import numpy as np
import time

'''
Dataset: MNIST
Time: 285s
Acuracy: 99%
'''


def loadData(fileName):
    print('start reading file')

    data = np.genfromtxt(fileName, delimiter=',')
    Label = data[:, 0]
    X = data[:, 1:] / 255

    return X, Label


def dist(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))


def getClosest(trainDataMat, trainLabelMat, x, topK):
    '''
    predict the class of given instance by selecting topK number of
    closest nodes and following the majority votes
    :param trainDataMat: training X
    :param trainLabelMat: training label
    :param x: a test instance
    :param topK: choice of K can have a huge impact on the accuracy, see chapter 3.2.3
    :return: class prediction
    '''

    # create a list containing distance of all nodes to the given instance
    distList = [0] * len(trainLabelMat)
    for i in range(len(trainDataMat)):
        x1 = trainDataMat[i]
        curDist = dist(x1, x)
        distList[i] = curDist

    # argsort：numpy list sorting algorithm
    # example
    #   >>> x = np.array([3, 1, 2])
    #   >>> np.argsort(x)
    #   array([1, 2, 0])
    # the algorithm returns the index of sorted elements, NOT the ELE itself !!!
    # now we take first topK number of votes
    topKList = np.argsort(np.array(distList))[:topK]

    # iterate through the vote list and accumulate the INDEX votes, the prediction should be the
    # class at INDEX with most votes
    labelList = [0] * 10
    for index in topKList:
        labelList[int(trainLabelMat[index])] += 1
    return labelList.index(max(labelList))

    # However the efficiency is incredibly unpromising, the process of sorting the whole distance list would be
    # considerably time-consuming (NLog(N)). Further improvement would be to implement KD tree, an advanced data
    # structure introduced in Chapter 3.3 with facilitate the CLOSEST node search in Log(N).


def model_test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, topK):
    print('start test')
    # transform data into matrix to enhance efficiency
    trainDataMat = np.matrix(trainDataArr);
    trainLabelMat = np.matrix(trainLabelArr).T
    testDataMat = np.matrix(testDataArr);
    testLabelMat = np.matrix(testLabelArr).T

    errorCnt = 0
    # since testing all dataset would be a night long, we manually choose first 200 data to manifest the algorithm
    # for i in range(len(testDataMat)):
    for i in range(200):
        # print('test %d:%d'%(i, len(trainDataArr)))
        print('test %d:%d' % (i, 200))
        # 读取测试集当前测试样本的向量
        x = testDataMat[i]
        # 获取预测的标记
        y = getClosest(trainDataMat, trainLabelMat, x, topK)
        # 如果预测标记与实际标记不符，错误值计数加1
        if y != testLabelMat[i]: errorCnt += 1

    # 返回正确率
    return 1 - (errorCnt / len(testDataMat))


if __name__ == '__main__':
    start = time.time()

    # 获取训练集
    trainDataArr, trainLabelArr = loadData('../../../dataset/mnist_train.csv')
    # 获取测试集
    testDataArr, testLabelArr = loadData('../../../dataset/mnist_test.csv')

    # trainDataArr, trainLabelArr, testDataArr, testLabelArr = genData(1000, 3, 2)
    # 计算测试集正确率
    accur = model_test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, 20)
    # 打印正确率
    print('accur is:%d' % (accur * 100), '%')

    end = time.time()
    # 显示花费时间
    print('time span:', end - start)
