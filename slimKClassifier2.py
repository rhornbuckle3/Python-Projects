import numpy as np
import pandas as pd
import math as mt
import random
import scipy.io
sampData=pd.read_csv("K_Means_Data.csv")
sampData=sampData.T
sampData=sampData.values
def clusterONE(kInput):
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
def clusterAssignLloyd(centerOne,centerTwo):
    kIndex=np.array((centerOne,centerTwo))
    labels=np.array(np.zeros(sampData.shape[1]))
    diffArray=np.copy(np.zeros(kIndex.shape[0]))
    for i in range(0,sampData.shape[1]):
        for j in range(0,kIndex.shape[0]):
            diffArray[j]=np.linalg.norm(sampData[:,i]-sampData[:,kIndex[j]],2)
        labels[i]=np.argmin(diffArray)
    modData=np.array(sampData)
    kThresher=np.array(np.zeros((modData.shape[0],kIndex.shape[0])))
    while(True):
        kIndexLocat=np.array(np.zeros((modData.shape[0],kIndex.shape[0])))
        #calculating new center off of average of label owners
        avgCalcVec=np.array(modData[:,0])
        avgCalcVec=avgCalcVec.reshape((-1,1))
        kKeeper=np.asscalar(labels[0])
        for j in range(0,kIndex.shape[0]):
            for i in range(1,labels.shape[0]):
                if(labels[i]==kKeeper):
                    avgCalcVec=np.append(avgCalcVec,modData[:,i].reshape((-1,1)),axis=1)
            for l in range(0,modData.shape[0]):
                        kIndexLocat[l,j]=np.sum(avgCalcVec[l,:])/avgCalcVec.shape[1]
            for i in range(0,labels.shape[0]):
                if(labels[i]!=kKeeper):
                    kKeeper=np.asscalar(labels[i])
        for i in range(0,sampData.shape[1]):
            for j in range(0,kIndex.shape[0]):
                diffArray[j]=np.linalg.norm(sampData[:,i]-kIndexLocat[:,j],2)
            labels[i]=np.argmin(diffArray)
        checker=0
        for i in range(0,modData.shape[0]):
            if((abs(kThresher[i,0]-kIndexLocat[i,0])<10e-4)&(abs(kThresher[i,1]-kIndexLocat[i,1])<10e-4)):
                checker=checker+1
        kThresher=np.array(kIndexLocat)
        if(checker==sampData.shape[0]):
            return labels
def lloydsCenterSpec(kPointOne,kPointTwo):
    modData=np.array(sampData)
    kIndex=np.array(np.zeros(2))
    labels=np.array(np.zeros(sampData.shape[1]))
    diffArray=np.copy(np.zeros(kIndex.shape[0]))
    kPointOne=kPointOne.reshape((-1,1))
    kPointTwo=kPointTwo.reshape((-1,1))
    kPointOne=np.append(kPointOne,kPointTwo,axis=1)
    for i in range(0,sampData.shape[1]):
            for j in range(0,kIndex.shape[0]):
                diffArray[j]=np.linalg.norm(sampData[:,i]-kPointOne[:,j],2)
            labels[i]=np.argmin(diffArray)
    kThresher=np.array(np.zeros((modData.shape[0],kIndex.shape[0])))
    while(True):
        kIndexLocat=np.array(np.zeros((modData.shape[0],kIndex.shape[0])))
        #calculating new center off of average of label owners
        avgCalcVec=np.array(modData[:,0])
        avgCalcVec=avgCalcVec.reshape((-1,1))
        kKeeper=np.asscalar(labels[0])
        for j in range(0,kIndex.shape[0]):
            for i in range(1,labels.shape[0]):
                if(labels[i]==kKeeper):
                    avgCalcVec=np.append(avgCalcVec,modData[:,i].reshape((-1,1)),axis=1)
            for l in range(0,modData.shape[0]):
                        kIndexLocat[l,j]=np.sum(avgCalcVec[l,:])/avgCalcVec.shape[1]
            for i in range(0,labels.shape[0]):
                if(labels[i]!=kKeeper):
                    kKeeper=np.asscalar(labels[i])
        for i in range(0,sampData.shape[1]):
            for j in range(0,kIndex.shape[0]):
                diffArray[j]=np.linalg.norm(sampData[:,i]-kIndexLocat[:,j],2)
            labels[i]=np.argmin(diffArray)
        checker=0
        for i in range(0,modData.shape[0]):
            if((abs(kThresher[i,0]-kIndexLocat[i,0])<10e-4)&(abs(kThresher[i,1]-kIndexLocat[i,1])<10e-4)):
                checker=checker+1
        kThresher=np.array(kIndexLocat)
        if(checker==sampData.shape[0]):
            return labels
def error(labels,truthFile):
    truthData=pd.read_csv(truthFile)
    truthData=truthData.T
    truthData=truthData.values
    iterate=0
    for i in range(0,truthData.shape[1]):
        if(labels[i]!=truthData[0,i]):
            iterate=iterate+1
    errorONE=(iterate/truthData.shape[1])*100
    iterate=0
    for i in range(0,labels.shape[0]):
        if(labels[i]==0):
            labels[i]=1
        else:
             if(labels[i]==1):
                labels[i]=0
    for i in range(0,truthData.shape[1]):
        if(labels[i]!=truthData[0,i]):
            iterate=iterate+1
    errorTwo=(iterate/truthData.shape[1])*100
    if(errorONE>errorTwo):
         return errorTwo
    else:
        return errorOne
def randomCenters(kInput):
    #not scalable yet
    ranCenOne=random.choice(range(0,sampData.shape[1]))
    ranCenTwo=random.choice(range(0,sampData.shape[1]))
    return ranCenOne,ranCenTwo
def trueCenterLoad():
    trueKcen=scipy.io.loadmat("mu_init.mat")
    trueKcen=trueKcen['mu_init']
    trueKcen=np.array(trueKcen)
    return trueKcen[:,0],trueKcen[:,1]
def simpleClass(centerOne,centerTwo):
    kIndex=np.array((centerOne,centerTwo))
    labels=np.array(np.zeros(sampData.shape[1]))
    diffArray=np.copy(np.zeros(kIndex.shape[0]))
    for i in range(0,sampData.shape[1]):
        for j in range(0,kIndex.shape[0]):
            diffArray[j]=np.linalg.norm(sampData[:,i]-sampData[:,kIndex[j]],2)
        labels[i]=np.argmin(diffArray)
    return labels
centerOne,centerTwo=clusterONE(2)
print(error(simpleClass(centerOne,centerTwo),"K_Means_Truth.csv"))
#with my algorithm for determining cluster centers and then assigning every sample to its closest center
#centerOne,centerTwo=trueCenterLoad()
#print(error(lloydsCenterSpec(centerOne,centerTwo),"K_Means_Truth.csv"))
#^with the true centers
#centerOne,centerTwo=randomCenters(2)
#print(error(clusterAssignLloyd(centerOne,centerTwo),"K_Means_Truth.csv"))
#^with random cluster centers
#implement iterative 'aggressive' logistic regression classifier for extra credit
#example
