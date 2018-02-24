import numpy as np
import pandas as pd
import math as mt
import numpy.random 
import matplotlib.pyplot as plt
sampData=pd.read_csv("K_Means_Data.csv")
sampData=sampData.T
sampData=sampData.values
def clusterONE(kInput):
    kNum=1
    norman=np.array(np.zeros((sampData.shape[0],sampData.shape[1]-1)))
    for i in range(0,sampData.shape[1]-1):
        norman[:,i]=sampData[:,0]-sampData[:,i+1]
    normanNorms=np.array(np.zeros(norman.shape[1]))
    for i in range(0,norman.shape[1]):
        normanNorms[i]=np.linalg.norm(norman[:,i],2)
    norman=np.append(norman,normanNorms.reshape((1,-1)),axis=0)
    normSort=np.copy(norman[:,norman[norman.shape[0]-1,:].argsort()])
    normDiff=np.array(np.zeros((normSort.shape[0]-1,normSort.shape[1]-1)))
    #^the vector for the difference between the differences, will be one smaller, remember to increment by one for matching in the future
    for i in range(0,normSort.shape[1]-1):
        normDiff[:,i]=normSort[0:normSort.shape[0]-1,i]-normSort[0:normSort.shape[0]-1,i+1]
    normAvg=np.array(np.zeros(normDiff.shape[1]-mt.floor(normDiff.shape[1]/(kInput+1))))
    #^the vector of average change in difference norm in groups of size n/(k+1)
    normsFin=np.array(np.zeros(normDiff.shape[1]))
    for i in range(0,normDiff.shape[1]):
        normsFin[i]=np.linalg.norm(normDiff[:,i],2)
    for i in range(0,normAvg.shape[0]):
        normAvg[i]=np.sum(normsFin[i:i+mt.floor(sampData.shape[1]/(kInput+1))])
    kOne=np.argmin(normAvg)
    kOneIndex=np.where(normanNorms==normSort[normSort.shape[0]-1,1+kOne+mt.floor(mt.floor(sampData.shape[1]/(kInput+1))/2)])
    kOneIndex=np.asscalar(kOneIndex[0])
    #now we repeat for the next k-1 points
    #remove the bottom (kNum-1)N/kInput of the difference sort for each new point to avoid finding the same cluster
    #need to make the following 20 lines of code scalable, shouldn't be too hard
    norman=np.array(np.zeros((sampData.shape[0],sampData.shape[1]-1)))
    for i in range(0,kOneIndex):
        norman[:,i]=sampData[:,kOneIndex]-sampData[:,i]
    for i in range(kOneIndex+1,sampData.shape[1]-1):
        norman[:,i]=sampData[:,kOneIndex]-sampData[:,i]
    for i in range(0,norman.shape[1]):
        normanNorms[i]=np.linalg.norm(norman[:,i],2)
    norman=np.append(norman,normanNorms.reshape((1,-1)),axis=0)
    normSort=np.copy(norman[:,norman[norman.shape[0]-1,:].argsort()])
    normDiff=np.array(np.zeros((normSort.shape[0]-1,normSort.shape[1]-1)))
    for i in range(0,normSort.shape[1]-1):
        normDiff[:,i]=normSort[0:normSort.shape[0]-1,i]-normSort[0:normSort.shape[0]-1,i+1]
    normAvg=np.array(np.zeros(normDiff.shape[1]-mt.floor(normDiff.shape[1]/(kInput+1))))
    for i in range(0,normDiff.shape[1]):
        normsFin[i]=np.linalg.norm(normDiff[:,i],2)
    for i in range(0,normAvg.shape[0]):
        normAvg[i]=np.sum(normsFin[i:i+mt.floor(sampData.shape[1]/(kInput+1))])
    kOne=np.argmin(normAvg[range(mt.floor(sampData.shape[1]/(kInput+1)),normAvg.shape[0])])
    kTwoIndex=np.where(normanNorms==normSort[normSort.shape[0]-1,1+kOne+mt.floor(mt.floor(sampData.shape[1]/(kInput+1))/2)])
    kTwoIndex=np.asscalar(kTwoIndex[0])
    return kOneIndex,kTwoIndex
def clusterAssign(kInput):
    centerOne,centerTwo=clusterONE(kInput)
    kIndex=np.array((centerOne,centerTwo))
    labels=np.array(np.zeros(sampData.shape[1]))
    diffArray=np.copy(np.zeros(kIndex.shape[0]))
    for i in range(0,sampData.shape[1]):
        for j in range(0,kIndex.shape[0]):
            diffArray[j]=np.linalg.norm(sampData[:,i]-sampData[:,kIndex[j]],2)
        labels[i]=np.argmin(diffArray)
    return labels
    
    #After this, things get kinda stupid. A bad first attempt
    threshold=True
    clusterCenters=np.array(np.zeros((sampData.shape[0],kIndex.shape[0])))
    modData=np.array(sampData)
    threshCheck=0
    while(threshold):
        kIndexLocat=np.array(np.zeros((modData.shape[0],kIndex.shape[0])))
        #calculating new center off of average of label owners
        for j in range(0,kIndex.shape[0]):
            avgCalcVec=None
            for i in range(0,labels.shape[0]):
                if(labels[i]==j):
                    if(type(avgCalcVec)=='numpy.ndarray'):
                        avgCalcVec=np.append(avgCalcVec,modData[:,i],axis=1)
                    else:
                        avgCalcVec=np.array(modData[:,i])
            for l in range(0,modData.shape[0]):
                    if(type(avgCalcVec)=='numpy.ndarray'):
                        print("here")
                        #THIS IS THE PROBLEM, IT NEVER GETS TO HERE
                        #solution on whiteboard, remember to put the whole thing in a loop cuz scale
                        kIndexLocat[l,j]=np.sum(avgCalcVec[l,:])/avgCalcVec.shape[1]
        for i in range(0,sampData.shape[1]):
            for j in range(0,kIndex.shape[0]):
                diffArray[j]=np.linalg.norm(sampData[:,i]-kIndexLocat[:,j],2)
            labels[i]=np.argmin(diffArray)
            print(labels[i])

        print(threshCheck)
        threshCheck=threshCheck+1
        if(threshCheck==2):
            threshold=False  
        #build actual threshold check, if average vector does nor change after an iteration, stop
        #newClusterCenter
    #print(labels)
    #print(kIndex)
    return labels
def error(labels):
    truthData=pd.read_csv("K_Means_Truth.csv")
    truthData=truthData.T
    truthData=truthData.values
    iterate=0
    for i in range(0,truthData.shape[1]):
        if(labels[i]!=truthData[0,i]):
            iterate=iterate+1
    return (iterate/truthData.shape[1])*100
#print(clusterONE(2))
#print(clusterAssign(2))
#clusterAssign(2)
print(error(clusterAssign(2)))
#implement iterative 'aggressive' logistic regression classifier for extra credit

