#fullstackClustering
import numpy as np
import pandas as pd
import math as mt
import random
import scipy.io
sampData=pd.read_csv("K_Means_Data.csv")
sampData=sampData.T
sampData=sampData.values
def clusterHOME(kInput):
    norman=np.array(np.zeros((sampData.shape[0],sampData.shape[1]-1)))
    for i in range(0,sampData.shape[1]-1):
        norman[:,i]=sampData[:,0]-sampData[:,i+1]
    normanNorms=np.array(np.zeros(norman.shape[1]))
    for i in range(0,norman.shape[1]):
        normanNorms[i]=np.linalg.norm(norman[:,i],2)
    norman=np.append(norman,normanNorms.reshape((1,-1)),axis=0)
    normSort=np.copy(norman[:,norman[norman.shape[0]-1,:].argsort()])
    #switch over to working exclusively with a vector of the sorted indices
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
    #^Return a vector instead
def recFunctONE(i,iBeta,xSet):
    #this is the actual classifier
    xTemp=xSet[:,i].reshape((-1,1))
    return 1/(1+mt.e**np.matmul(np.transpose(-iBeta),xTemp))
def gradientProb(iBeta,xSet,ySet):
    gradLProb=0
    for i in range(0,xSet.shape[1]):
        gradLProb+=np.multiply(ySet[:,i]-recFunctONE(i,iBeta,xSet),xSet[:,i].reshape((-1,1)))
    return gradLProb
def gradientAscent(eta,tol,iBeta,xSet,ySet):
    while(True):
        grad=np.array(iBeta)
        print (grad)
        np.add(iBeta,eta*gradientProb(iBeta,xSet,ySet),out=iBeta)
        if((np.argmax(abs(iBeta-grad)))<tol):
            return iBeta
#construct the model for deploying log regression, details on whiteboard