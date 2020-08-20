# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 06:45:57 2019

@author: Samama
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
No_of_iterations=np.zeros(1)
N=100
I=1000
eta=0.1
unit_step = lambda z: -1 if z < 0 else 1
for k in range(I):  
    X=np.random.uniform(-1,1,size=(N,3))
    X[:,2]=1
    a,b=np.random.uniform(-1,1,size=(2,2))
    target_slope,target_intercept=np.polyfit(a,b,1)
    Y=np.zeros(len(X))
    for i  in range(len(X)):
        if (X[i,1]-target_slope*X[i,0]-target_intercept)>=0:
            Y[i]=1
        else:
            Y[i]=-1        
    w=np.zeros(3)
    PQ=[]
    errors=[]
    er=np.zeros(1)
    y=np.zeros(len(Y))
    stop_loop=1
    single_loop_iterations=0
    while stop_loop >0:
        for ss in range(len(Y)):
            if Y[ss]!=y[ss]:
                PQ=np.append(PQ,ss)
        if len(PQ)==0:
                stop_loop=-1
        else:    
            single_loop_iterations +=1
            pq=int(np.random.choice(PQ))
            x =X[pq]
            ytarget=Y[pq]
            result=np.dot(w,x)
            error=ytarget-unit_step(result)
            w +=  eta*error * x
            for r in range(len(Y)):
                dot_product=np.dot(w,X[r])
                y[r]=unit_step(dot_product)
        errors.append(error)
        PQ=[]
    No_of_iterations=np.append(No_of_iterations,single_loop_iterations)
    

print("Iterations perceptron took to estimate target function:",np.mean(No_of_iterations))

c=(-1,1)
perceptron_slope=-w[0]/w[1]
perceptron_intercept=-w[2]/w[1]
points_on_perceptron_estimator=(-1*perceptron_slope+perceptron_intercept,perceptron_slope+perceptron_intercept)
points_on_target_function=(-1*target_slope+target_intercept,target_slope+target_intercept)

Target_Function=plt.figure(1)
ay=Target_Function.add_subplot(111)
ay.scatter(X[:,0],X[:,1],c=Y,cmap=cm.coolwarm)
TF=ay.plot(c,points_on_target_function,color="orange",label="Target Function")
ay.set_xlabel("X1")
ay.set_ylabel("X2")
Target_handles,Target_labels=ay.get_legend_handles_labels()
ay.legend(Target_handles,Target_labels,loc="best")
ay.set_title("Target Function Classification")
Target_Function.savefig("Target_Function-2.png")
Comparison=plt.figure(1)
ax = Comparison.add_subplot(111)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_title("Perceptron estimation of Target Function")
ax.scatter(X[:,0],X[:,1],c=Y,cmap=cm.coolwarm)
CPE=ax.plot(c,points_on_perceptron_estimator,color="green",label="Perceptron Estimator")
CTF=ax.plot(c,points_on_target_function,color="orange",label="Target Function")
Comparison_handles,Comparison_labels=ax.get_legend_handles_labels()
ax.legend(Comparison_handles,Comparison_labels,loc="best")
Comparison.savefig("Comaprison_between_target_function_and_perceptron_estimator-2.png")
