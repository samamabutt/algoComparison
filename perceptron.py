# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:48:42 2019

@author: Samama
"""
import pickle
import numpy as np

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


label = []
img = []


    
        
    

 


class Perceptron(object):

    def __init__(self, no_of_inputs=2, threshold=200, learning_rate=0.001):
        
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = [0 for i in range(785)]#*(total)#np.zeros(no_of_inputs + 1)
        #self.weights = self.weights.reshape(784,)   
        
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


def verification(check,sample):
    
    print  "Input:", check

    for i in range(200):
       if trainLabels[i][0].astype(int) == check:
           label.append(i)
    for j in range(len(label)):
        img.append(trainImgs[label[j]])
    perceptron = Perceptron(2)
    perceptron.train(img, label)
    result= perceptron.predict(testImgs[sample])
    print "Prediction:",result
                
    errorFound = []               
    
    for i in range(200):
            
        result= perceptron.predict(testImgs[i])
        
        #print "imgae no.",i,"",sample,"=",result
        if testLabels[i].astype(int) == check and result == 1:
            
            errorFound.append(1)
            
        elif testLabels[i].astype(int) != check and result == 0:
            
            errorFound.append(1)
            
        else:
            
            errorFound.append(0)
                    
                    
           
    newError = 0.00
    totalExamples = len(errorFound) + 0.00            
    for element in errorFound:
        if element ==0:
            newError =  newError + 1            
    print "Accuracy:",100 - (newError/totalExamples)*100   



verification(0,3)
verification(1,2)
verification(2,1)
verification(3,18)
verification(4,4)
verification(5,8)
verification(6,88)
verification(7,0)
verification(8,61)
verification(9,99)
