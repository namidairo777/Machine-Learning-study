'''
2017.1.24
k-nearest neighbor
'''

from numpy import *
from os import listdir
import operator

def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	'''
	Simple classifier 
	inX: input vector for classification
	dataSet: training sample
	labels: label vector
	k: number of nearest neighbors
	return: 
	'''
	dataSetSize = dataSet.shape[0] # Row num

	# Calculate distance - euclidean distance
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5

	sortedDistIndicies = distances.argsort() # argsort: return the indices that would sort an array
	classCount = {}

	# Choose k num nearest neighbors
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

	# Sort
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	'''
	File to matrix
	filename: name
	'''
	fr =open(filename)
	arrayOfLines = fr.readlines() # file contest to a list
	numberOfLines = len(arrayOfLines)

	returnMat = zeros((numberOfLines, 3))

	classLabelVector = []
	index = 0
	for line in arrayOfLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index, :] = listFromLine[0:3]
		classLabelVector.append(listFromLine[-1])
		index += 1

	return returnMat, classLabelVector


def autoNorm(dataSet):
	'''
	Normalization
	dataSet: training sample
	'''
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m, 1))
	normDataSet = normDataSet/tile(ranges, (m, 1))
	return normDataSet, ranges, minVals

def datingClassTest():
	'''
	Test dating classifier
	'''
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errotCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
		if classifierResult != datingLabels[i]: errotCount += 1.0
	print "Error rate is %f" % (errotCount/float(numTestVecs))	

def classifyPerson():
	'''
	Classifier for person you would probably like
	'''
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentGames = float(raw_input("percentage of time spent playing video games?"))
	ffMiles = float(raw_input("flier miles earned per year?"))
	icecream = float(raw_input("liters of icecream consumed per year?"))
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentGames, icecream])
	classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
	print "you will probably like this person:", resultList[int(classifierResult) - 1]

def img2vector(filename):
	'''
	Image to vector
	'''
	returnVector = zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVector[0, 32*i+j] = int(lineStr[j])
	return returnVector

def handwritingClassTest():
	'''
	Classifier for handwriting
	'''
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m, 1024))
	for i in range(m):
		filenameStr = trainingFileList[i]
		fileStr = filenameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i, :] = img2vector('trainingDigits/%s' % filenameStr)

	testFileList = listdir('testDigits')
	errotCount = 0.0
	mTest = len(testFileList)

	for i in range(mTest):
		filenameStr = testFileList[i]
		fileStr = filenameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % filenameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

		if classifierResult != classNumStr: errotCount += 1.0

	print "total number of errors:%d" % errotCount
	print "error rate: %f" % (errotCount/float(mTest))
