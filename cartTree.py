import numpy as np
import random


'''
根据基尼系数构建一棵CART分类树
'''
# 准备数据


def loadDataSet():
    # # 最后一列的是 否表示是否欠款，即将预测的的属性
    # dataSet = [['是', '单身', 125, '未欠款'], ['否', '已婚', 100, '未欠款'], ['否', '单身', 70, '未欠款'],
    #             ['是', '已婚', 120, '未欠款'], ['否', '离异', 95, '欠款'], ['否', '已婚', 60, '未欠款'],
    #             ['是', '离异', 220, '未欠款'], ['否', '单身', 85, '欠款'], ['否', '已婚', 75, '未欠款'],
    #             ['否', '单身', 90, '欠款']]
    # labels = ['有房者', '婚姻情况', '年收入']
    # # 0表示离散值，1表示连续值
    # flags = [0, 0, 1]
    # dataSet = [['晴', '热', '高', '否', 'No'], ['晴', '热', '高', '是', 'No'], ['阴', '热', '高', '否', 'Yes'], ['雨', '适中', '高', '否', 'Yes'],
    #             ['雨', '凉爽', '正常', '否', 'Yes'], ['雨', '凉爽', '正常', '是', 'No'], ['阴', '凉爽', '正常', '是', 'Yes'], ['晴', '适中', '高', '否', 'No'],
    #             ['晴', '凉爽', '正常', '否', 'Yes'], ['雨', '适中', '正常', '否', 'Yes'], ['晴', '适中', '正常', '是', 'Yes'], ['阴', '适中', '高', '是', 'Yes'],
    #             ['阴', '热', '正常', '否', 'Yes'], ['雨', '适中', '高', '是', 'No']]
    # labels = ['天气', '温度', '湿度', '多风']
    # # 0表示离散值，1表示连续值
    # flags = [0, 0, 0, 0]
	dataSet = []
	path = r'F:/LearningRelated/UCI_DataSet/iris/iris.data'
	with open(path, encoding='utf-8') as file:
		for row in file.readlines():
			row = row.replace('\n', '').split(',')
			for i in range(len(row) - 1):
				row[i] = float(row[i])
			dataSet.append(row)
		labels = ['萼片长', '萼片宽', '花瓣长度', '花瓣宽度']
		flags = [1, 1, 1, 1]
		random.shuffle(dataSet)
		return dataSet[:], labels, flags


# 计算Gini系数
def calcGini(dataSet):
	'''
	:param dataSet: 待计算Gini系数的数据集
	:return: 当前数据集的Gini值
	'''
	length = len(dataSet)
	classLabel = {}
	for ele in dataSet:
		if ele[-1] not in classLabel.keys():
			classLabel[ele[-1]] = 0
		classLabel[ele[-1]] += 1
	gini = 0
	for key in classLabel:
		gini += np.square(classLabel[key] / length)
	return 1-gini


# 计算条件Gini系数
def calcConditionGini(leftDataSet, rightDataSet):
	leftGini = len(leftDataSet) / len(dataSet) * calcGini(leftDataSet)
	rightGini = len(rightDataSet) / len(dataSet) * calcGini(rightDataSet)
	conGini = leftGini + rightGini
	return conGini


# 划分数据集
def splitDataSet(dataSet, index, value, flag):
	leftDataSet = []
	rightDataSet = []
	for ele in dataSet:
		if flag == 0:  # 该列数据是离散的
			if ele[index] == value:
				leftDataSet.append(ele)
			else:
				rightDataSet.append(ele)
		else:  # 该列数据是连续型
			if ele[index] <= value:
				leftDataSet.append(ele)
			else:
				rightDataSet.append(ele)
	# 如果是离散值，左子树不需要包含该列数据
	if flag == 0:
		for i in range(len(leftDataSet)):
			leftDataSet[i] = leftDataSet[i][:index] + leftDataSet[i][index+1:]
	return leftDataSet, rightDataSet


# 选择基尼系数最小的特征和对应的特征值
def chooseBestFeatAndVale(dataSet, flags):
	bestGini = 1.0
	bestFeat = 0
	bestFeatValue = dataSet[0][0]
	# 统计一个样本的特征数
	featCount = len(dataSet[0]) - 1
	for i in range(featCount):
		currGini = 1.0
		# 找出当前特征所包含的特征值
		featValue = [ele[i] for ele in dataSet]
		featValue = list(set(featValue))
		# 判断当前特征是连续值还是离散值,0表示离散，1表示连续
		if flags[i] == 0:
			# 若该特征只有两个或以下的取值
			if len(featValue) <= 2:
				leftDataSet, rightDataSet = splitDataSet(dataSet, i, featValue[0], flags[i])
				currGini = calcConditionGini(leftDataSet, rightDataSet)
				currValue = featValue[0]
			else:
				for value in featValue:
					leftDataSet, rightDataSet = splitDataSet(dataSet, i, value, flags[i])
					multiGini = calcConditionGini(leftDataSet, rightDataSet)
					if multiGini < currGini:
						currGini = multiGini
						currValue = value
			# print(currGini, currValue)
		else:  # 处理连续型数值
			featValue.sort()
			splitPoint = [(featValue[i]+featValue[i+1])/2.0 for i in range(len(featValue) - 1)]
			for point in splitPoint:
				leftDataSet, rightDataSet = splitDataSet(dataSet, i, point, flags[i])
				multiGini = calcConditionGini(leftDataSet, rightDataSet)
				if multiGini < currGini:
					currGini = multiGini
					currValue = point
			# print(currGini, currValue)

		if currGini < bestGini:
			bestGini = currGini
			bestFeat = i
			bestFeatValue = currValue
	return bestFeat, bestFeatValue

# 选择当前数据集占大多数类别的类
def chooseMajorClass(dataSet):
	classCount = {}
	for ele in dataSet:
		if ele[-1] not in classCount:
			classCount[ele[-1]] = 0
		classCount[ele[-1]] += 1
	return sorted(list(zip(classCount.values(), classCount.keys())))[-1][-1]


# 创建CART分类树
def createCartTree(dataSet, giniThreshold, flags, labels):
	# 停止条件，当当前数据集的基尼系数小于阈值时，则停止/或当前数据集无特征时，则停止
	if calcGini(dataSet) <= giniThreshold or len(dataSet[0]) == 1:
		return chooseMajorClass(dataSet)
	# 找到当前数据集最好分裂节点及节点值
	bestFeat, bestFeatValue = chooseBestFeatAndVale(dataSet, flags)
	bestFeatName = labels[bestFeat]
	cartTree = {bestFeatName: {}}
	# 将最优节点拆分为两棵子树构建
	leftDataSet, rightDataSet = splitDataSet(dataSet, bestFeat, bestFeatValue, flags[bestFeat])
	# 依据该列的特征值的个数进行划分，若值个数为2，则左右子集都除去该列的值、若值的个数大于2，则值处理左子集
	# 处理离散值数据集
	leftFlags, rightFlags = flags[:], flags[:]  # 正确复制列表的内容，便于后续递归使用该值
	leftLabels, rightLabels = labels[:], labels[:]
	if flags[bestFeat] == 0:
		featValue = [ele[bestFeat] for ele in dataSet]
		featValue = list(set(featValue))
		if len(featValue) <= 2:
			for i in range(len(rightDataSet)):
				rightDataSet[i] = rightDataSet[i][:bestFeat] + rightDataSet[i][bestFeat+1:]
			del rightFlags[bestFeat]
			del rightLabels[bestFeat]
		# 不论左右子树，左子树都会删除相应的特征列
		del leftFlags[bestFeat]
		del leftLabels[bestFeat]
		rightFeatValue = list(set(featValue) - set([bestFeatValue]))
		cartTree[bestFeatName][str([bestFeatValue])] = createCartTree(leftDataSet, giniThreshold, leftFlags, leftLabels)
		cartTree[bestFeatName][str(rightFeatValue)] = createCartTree(rightDataSet, giniThreshold, rightFlags, rightLabels)
	else:
		cartTree[bestFeatName]['<='+str(bestFeatValue)] = createCartTree(leftDataSet, giniThreshold, leftFlags, leftLabels)
		cartTree[bestFeatName]['>'+str(bestFeatValue)] = createCartTree(rightDataSet, giniThreshold, rightFlags, rightLabels)
	return cartTree


# 主函数
if __name__ == '__main__':
	# loadDataSet()
    dataSet, labels, flags = loadDataSet()
    # calcGini(dataSet)
    # chooseBestFeatAndVale(dataSet, flags)
    # print(splitDataSet(dataSet, 0, '是'))
    # chooseMajorClass(dataSet)
    print(createCartTree(dataSet, 0.1, flags, labels))