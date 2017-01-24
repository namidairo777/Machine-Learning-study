from numpy import *
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
		classLabelVector.append(int(listFromLine[-1]))
		index += 1

	return returnMat, classLabelVector
	

