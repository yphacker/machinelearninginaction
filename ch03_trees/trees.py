# coding=utf-8
# author=yphacker


from utils.tree_utils import createTree
import treePlotter


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def main():
    dataSet, labels = createDataSet()
    myTree = treePlotter.retrieveTree()
    print classify(myTree, labels, [1, 0])
    print classify(myTree, labels, [1, 1])


if __name__ == "__main__":
    main()
