from __future__ import division

import numpy as np

import pickle



#trainData = np.loadtxt("mnist_train.csv", delimiter=",")
#
#testData = np.loadtxt("mnist_test.csv", delimiter=",")

    

def mnistData():
    
    
    
#    fac = 255  *0.99 + 0.01
#    
#    trainImgs = np.asfarray(trainData[:, 1:]) / fac
#    
#    testImgs = np.asfarray(testData[:, 1:]) / fac
#    
#    trainLabels = np.asfarray(trainData[:, :1])
#    
#    testLabels = np.asfarray(testData[:, :1])
#    
    
#    with open("pickled_mnist.pkl", "w") as fh:
#        data = (trainImgs, 
#                testImgs, 
#                trainLabels,
#                testLabels,
#                trainLabelsOneHot,
#                testLabelsOneHot)
#        pickle.dump(data, fh)


    
   
    with open("pickled_mnist.pkl", "r") as fh:
            data = pickle.load(fh)
            
    getTrainImgs = data[0]
    
    getTestImgs = data[1]#1
    
    getTrainLabels = data[2]
    
    getTestLabels = data[3]#2
    
    getTrainLabelsOneHot = data[4]
    
    getTestLabelsOneHot = data[5]
    #From begining to 60,000(excluded)
    
    trainImgs = np.array(getTrainImgs[:60000], copy = True)
    
    trainLabels = np.array(getTrainLabels[:60000], copy = True)
    
    testImgs = np.array(getTestImgs[:60000], copy = True)
    
    testLabels = np.array(getTestLabels[:60000], copy = True)
    
    
    classifierNQ(0, 3, trainLabels, trainImgs, testImgs, testLabels)
    classifierNQ(1, 2, trainLabels, trainImgs, testImgs, testLabels)
    classifierNQ(2, 1, trainLabels, trainImgs, testImgs, testLabels)
    classifierNQ(3, 18, trainLabels, trainImgs, testImgs, testLabels)
    classifierNQ(4, 4, trainLabels, trainImgs, testImgs, testLabels)
    classifierNQ(5, 8, trainLabels, trainImgs, testImgs, testLabels)
    classifierNQ(6, 88, trainLabels, trainImgs, testImgs, testLabels)
    classifierNQ(7, 0, trainLabels, trainImgs, testImgs, testLabels)
    classifierNQ(8, 61, trainLabels, trainImgs, testImgs, testLabels)
    classifierNQ(9, 99, trainLabels, trainImgs, testImgs, testLabels)
   
    

    
    '''
     for i in range(100):
        print i, "=", getTestLabels[i].astype(int)
    0 = [7]
    1 = [2]
    2 = [1]
    3 = [0]
    4 = [4]
    11 = [6]
    18 = [3]
    7 = [9]
    8 = [5]
    9 = [9]
    61 = [8]
    '''
    
    
def classifierNQ(check, test, trainLabels, trainImgs, testImgs, testLabels):


    for i in range(len(trainLabels)):

        if (trainLabels[i][0].astype(int) == check):
            
            trainLabels[i] = 1
            
        else:
            
            trainLabels[i] = -1
    
    for i in range(len(testLabels)):

        if (testLabels[i][0].astype(int) == check):
            
            testLabels[i] = 1
            
        else:
            
            testLabels[i] = -1
    
    #Taking Transpose
    mTranspose = trainImgs.transpose()
    #Taking dot product of M with M Transpose
    mDotProduct = np.matmul(mTranspose , trainImgs)
    #Taking inverse of dot product of M with M Transpose
    inverse = np.linalg.pinv(mDotProduct)
    
    mTransposeDotProduct = np.matmul(mTranspose,trainLabels)
    
    normalEQ = np.matmul(inverse,mTransposeDotProduct)
    
    normalEQ = normalEQ.reshape(784,)
            
    prediction = np.dot(normalEQ,testImgs[test])
    
    
    print "Normal Equation Expected Value: ",check
    
    print "Normal Equation Predicted Value: ",prediction
    
    errorFound = []

    for i in range(200):
        
            sample = testImgs[i]
            
            result = np.dot(normalEQ, sample)
            
            #print "imgae no.",i,"",sample,"=",result
            
            if testLabels[i].astype(int) == check and result > 0:
                
                errorFound.append(1)
                
            elif testLabels[i].astype(int) != check and result < 0:
                
                errorFound.append(1)
                
            else:
                
                errorFound.append(0)
                
                
       
    newError = 0
                
    for element in errorFound:
        if element ==0:
            newError =  newError + 1            
    print "Accuracy:",100 - (newError/len(errorFound))*100    
   
    gradientDescent(check,trainImgs,normalEQ,testImgs,testLabels,test,trainLabels)

def gradientDescent(check,trainD,vec,tstImg,tstLbl,test,trainLbl):
    
    learnRate = 0.0001
    
    gd = 0.0000000000
    
    for j in range (200):
       
        for i in range(200):
            
            change = 0.00000000000000000000
            
            for s in range(200):
                
                error = 0.00000000000000
                
                actualResult = np.dot(vec, trainD[s])
                
                expecResult = trainLbl[s][0]
                
                error = expecResult - actualResult
                
                error = error * trainD[s][i]
                
                change = change + error
                
            gd = learnRate * change
            
            vec[i] = vec[i] + gd
            
    
    result = np.dot(vec, tstImg[test])
    
    print "Gradient Descent Expected Value:",check
    
    print "Gradient Descent prediction:", result
    
    errorFound = []

    for i in range(200):
        
            sample = tstImg[i]
            
            result = np.dot(vec, sample)
            
            #print "imgae no.",i,"",sample,"=",result
            
            if tstLbl[i].astype(int) == check and result > 0:
                
                errorFound.append(1)
                
            elif tstLbl[i].astype(int) != check and result < 0:
                
                errorFound.append(1)
                
            else:
                
                errorFound.append(0)
                
                
       
    newError = 0
                
    for element in errorFound:
        if element ==0:
            newError =  newError + 1            
    print "Gradient Descent Accuracy:",100 - (newError/len(errorFound))*100
    

    
    

#def classifierPercep(check, test, trainLabels, trainImgs, testImgs, testLabels):
#    
#    learningRate = 0.0001
#    
#    for i in range(len(trainLabels)):
#
#        if (trainLabels[i][0].astype(int) == check):
#            trainLabels[i] = 1
#        else:
#            trainLabels[i] = -1
#            
#    for i in range(len(testLabels)):
#
#        if (testLabels[i][0].astype(int) == check):
#            testLabels[i] = 1
#        else:
#            testLabels[i] = -1
#    
#    perceptron = [0 for i in range(784)]
#    
#    for j in range(100):
#        
#        for i in range(200):#784
#            
#            finalError = 0.00000000000
#            
#            for s in range(100):
#                
#                error = 0.000000
#                
#                actualValue = np.dot(perceptron,trainImgs[s])
#                
#                expectedValue = trainLabels[s][0]
#                
#                error = expectedValue - actualValue
#                
#                error = error * trainImgs[s][i]
#                
#                finalError = finalError + error
#                
#            totalChange = learningRate * finalError
#            
#            perceptron[i] = perceptron[i] + totalChange
#            
#            
#    perceptronPrediction = np.dot(perceptron,testImgs[test])
#    
#    print "Perceptron Expected Value: ", check
#    
#    print "Percepron Predicted Value: ", perceptronPrediction     
#        
#    errorFound = []
#
#    for i in range(200):
#        
#        sample = testImgs[i]
#        
#        result = np.dot(perceptron, sample)
#        
#        #print "imgae no.",i,"",sample,"=",result
#        if testLabels[i].astype(int) == check and result > 0:
#            
#            errorFound.append(1)
#            
#        elif testLabels[i].astype(int) != check and result < 0:
#            
#            errorFound.append(1)
#            
#        else:
#            
#            errorFound.append(0)
#                
#                
#       
#    newError = 0
#                
#    for element in errorFound:
#        if element ==0:
#            newError =  newError + 1            
#    print "Accuracy:",100 - (newError/len(errorFound))*100   

mnistData()


                