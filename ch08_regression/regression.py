# coding=utf-8
# author=yphacker


import numpy as np
from utils.lwlr_util import ridgeTest, rssError
from bs4 import BeautifulSoup


def scrapePage(inFile, outFile, yr, numPce, origPrc):
    fr = open(inFile)
    fw = open(outFile, 'a')
    soup = BeautifulSoup(fr.read())
    i = 1
    currentRow = soup.findAll('table', r="%d" % i)
    while (len(currentRow) != 0):
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde) == 0:
            print("item #%d did not sell" % i)
        else:
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')  # strips out $
            priceStr = priceStr.replace(',', '')  # strips out ,
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')  # strips out Free Shipping
            print("%s\t%d\t%s" % (priceStr, newFlag, title))
            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr, numPce, newFlag, origPrc, priceStr))
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)
    fw.close()


def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = range(m)
    errorMat = np.zeros((numVal, 30))  # create error mat 30columns numVal rows
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        np.random.shuffle(indexList)
        for j in range(m):  # create training set based on first 90% of values in indexList
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)  # get 30 weight vectors from ridge
        for k in range(30):  # loop over all of the ridge estimates
            matTestX = np.mat(testX);
            matTrainX = np.mat(trainX)
            meanTrain = np.mean(matTrainX, 0)
            varTrain = np.var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain  # regularize test with training params
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)  # test ridge results and store
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY))
            # print errorMat[i,k]
    meanErrors = np.mean(errorMat, 0)  # calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    # can unregularize to get model
    # when we regularized we wrote Xreg = (x-meanX)/np.var(x)
    # we can now write in terms of x not Xreg:  x*w/np.var(x) - meanX/np.var(x) +meanY
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0)
    varX = np.var(xMat, 0)
    unReg = bestWeights / varX
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ", -1 * sum(np.multiply(meanX, unReg)) + np.mean(yMat))


def main():
    pass


if __name__ == "__main__":
    main()
