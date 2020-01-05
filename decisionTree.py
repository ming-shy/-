import numpy as np
import os


# 获取训练集
def getTrainDataSet():
    # dataSet = [[1, 1, 'yes'], [1, 1, 'yes'],
    #            [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    # # 每个特征的使用的特征名
    # labels = ['no surfacing', 'flippers']
    dataSet = [['晴', '热', '高', '否', 'No'], ['晴', '热', '高', '是', 'No'], ['阴', '热', '高', '否', 'Yes'],
               ['雨', '适中', '高', '否', 'Yes'],
               ['雨', '凉爽', '正常', '否', 'Yes'], ['雨', '凉爽', '正常', '是', 'No'], ['阴', '凉爽', '正常', '是', 'Yes'],
               ['晴', '适中', '高', '否', 'No'],
               ['晴', '凉爽', '正常', '否', 'Yes'], ['雨', '适中', '正常', '否', 'Yes'], ['晴', '适中', '正常', '是', 'Yes'],
               ['阴', '适中', '高', '是', 'Yes'],
               ['阴', '热', '正常', '否', 'Yes'], ['雨', '适中', '高', '是', 'No']]
    labels = ['天气', '温度', '湿度', '多风']
    return dataSet, labels


# 计算给定数据集的香农熵
def calShannonEnt(dataset):
    # 统计数据集的个数
    counts = len(dataset)
    # 用来记录每一类的个数
    labelCounts = {}
    # 1.将数据集中标签按类别分开并统计其个数
    for ele in dataset:
        if ele[-1] not in labelCounts.keys():
            labelCounts[ele[-1]] = 0
        labelCounts[ele[-1]] += 1
    # 2.计算香农熵
    shannonEnt = 0.0
    for ele in labelCounts:
        p = labelCounts[ele] / counts
        shannonEnt -= p * np.log10(p)
    return shannonEnt


# 划分指定列【即某一特征列】为特定值【在特征列上会有不同的取值】的数据集
def splitDataSet(dataset, index, value):
    '''
    :param dataset: 初始数据集
    :param index: 指定列或指定的特征列
    :param value: 该特征列上指定的值
    :return: 指定列上为特定值的数据集
    '''
    specificDataSet = []
    for ele in dataset:
        if ele[index] == value:
            specificDataSet.append(ele)
    # 不包含指定列的数据集
    for i in range(len(specificDataSet)):
        specificDataSet[i] = specificDataSet[i][:index] + specificDataSet[i][index+1:]
    return specificDataSet


# 获取某一特征列的数据集
def getFeatColumn(dataset, index):
    featColumnData = []
    for ele in dataset:
        featColumnData.append([ele[index]])
    return featColumnData


# 选择最好的特征作为节点，根据信息增益（ID3算法）进行选择
def chooseBestFeature(dataset):
    # 数据集的熵
    baseEntro = calShannonEnt(dataset)
    # 最大的信息增益
    maxGain = 0.0
    # 初始特征列
    bestFeatColumn = 0
    # 1.确定一个样本有几个特征
    numFeatures = len(dataset[0]) - 1
    # 2.遍历所有特征列，并求出其在该特征条件下的熵，再求出其信息增益，找出最大信息增益作为需要的节点
    for i in range(numFeatures):
        # 对该特征列统计有多少取值
        featValueList = [ele[i] for ele in dataset]
        # 取出重复值
        featValueList = list(set(featValueList))
        conEntro = 0.0
        for featValue in featValueList:
            specfDataSet = splitDataSet(dataset, i, featValue)
            prob = len(specfDataSet) / len(dataset)
            conEntro += prob * calShannonEnt(specfDataSet)   # 在某一特征下的条件熵
        # 信息增益
        infoGain = baseEntro - conEntro
        # 当前特征列所对应的熵
        featEntro = calShannonEnt(getFeatColumn(dataset, i))
        # 信息增益率(C4.5中使用)
        infoGainRate = infoGain / featEntro
        # print(baseEntro, infoGain, featEntro, infoGainRate)
        # 选择最大的信息增益特征列的小标
        if infoGainRate > maxGain:
            maxGain = infoGainRate
            bestFeatColumn = i
    return bestFeatColumn  # 返回为【信息增益】最大的特征列的下标


# 选择出现类别较多的一类
def chooseMajorClass(classList):
    classCount = {}
    for ele in classList:
        if ele not in classCount:
            classCount[ele] = 0
        classCount[ele] += 1
    # 使用zip(),将字典键值对重组，并进行排序
    classCount = sorted(list(zip(classCount.values(), classCount.keys())), reverse=True)
    return classCount[0][-1]


# 创建决策树（暂时不进行剪枝）
def createDecisionTree(dataset, labels):
    classList = [ele[-1] for ele in dataset]
    # 第一个停止条件【若数据集中类标签只有一个】，即确定了叶节点
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 返回yes or no
    # 第二个停止条件【若使用完所有特征，仍不能确定所属类别】
    if len(dataset[0]) == 1:
        return chooseMajorClass(classList)
    # 获取信息增益最大的特征列的下标
    bestFeatColumn = chooseBestFeature(dataset)
    bestFeatLabel = labels[bestFeatColumn]
    # 使用字典来表示树，其中key表示节点，value表示边
    decisionTree = {bestFeatLabel: {}}
    # print(labels[bestFeatColumn])
    # 复制原有的label进行删除，避免后续递归使用出现错误
    newLabels = labels[:]
    del newLabels[bestFeatColumn]
    # 统计该列的类数
    featValue = [ele[bestFeatColumn] for ele in dataset]
    featValue = list(set(featValue))
    # 根据特征列和特征值划分数据集
    for value in featValue:
        currDataSet = splitDataSet(dataset, bestFeatColumn, value)
        decisionTree[bestFeatLabel][value] = createDecisionTree(currDataSet, newLabels)
    return decisionTree


# 保存训练好的决策树模型
def saveDecisionTree(inputTree):
    basePath = os.path.dirname(__file__)
    inputTree = str(inputTree)
    with open(basePath + '/tree.txt', 'w') as file:
        file.write(inputTree)
    print('=' * 10 + "决策树模型保存成功" + '=' * 10)


# 测试决策树模型
def testDecisionTree(labels, testData):
    base_path = os.path.dirname(__file__)
    classLabels = ['yes', 'no']
    with open(base_path+'/tree.txt') as file:
        treeModel = eval(file.read())  # 使用eval()将str-->dict,json.loads()也可将str-->dict
        nodeValue = list(treeModel.keys())[0]
        subTree = treeModel[nodeValue]
        # 决策树的高度最多为特征数+1
        for depth in range(len(labels)+1):
            currTree = subTree[testData[labels.index(nodeValue)]]
            if currTree not in classLabels:
               nodeValue = list(currTree.keys())[0]
               subTree = currTree[nodeValue]
            else:
                return currTree


if __name__ == '__main__':
    dataset, labels = getTrainDataSet()
    # print(calShannonEnt(dataset))
    # splitDataSet(dataset, 1, 0)
    # print(chooseBestFeature(dataset))
    # print(chooseBestFeature(dataset))
    # tree = createDecisionTree(dataset, labels)
    print(createDecisionTree(dataset, labels))
    # print(tree)
    # saveDecisionTree(tree)
    # print(testDecisionTree(labels, [1, 1]))