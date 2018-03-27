import numpy as np 
import math as mt 
import scipy.io as sio
import matplotlib.pyplot as plt
thumbData=sio.loadmat("thumb_data.mat")
print(thumbData.keys())
sampData=np.array(thumbData['X'])
yData=np.array(thumbData['mu'])
corrMat=np.array(np.zeros((sampData.shape[0],sampData.shape[1])))
avgMat=np.copy(corrMat)
yAvg=np.average(yData)
for j in range(0,64):
    for k in range(0,64):
        avgMat[j,k]=np.average(sampData[j,k,:])
        tempMatTop=np.array(np.zeros((1,122)))
        tempMatBotOne=np.array(np.zeros((1,122)))
        tempMatBotTwo=np.array(np.zeros((1,122)))
        for i in range(0,122):
            tempMatTop[0,i]=(sampData[j,k,i]-avgMat[j,k])*(yData[i]-yAvg)
            tempMatBotOne[0,i]=(sampData[j,k,i]-avgMat[j,k])**2
            tempMatBotTwo[0,i]=(yData[i]-yAvg)**2
        corrMat[j,k]=np.fabs(np.sum(tempMatTop)/(mt.sqrt(np.sum(tempMatBotOne)*np.sum(tempMatBotTwo))))

for i in range(0,64):
    for j in range(0,64):
        if(mt.isnan(corrMat[i,j])):
            corrMat[i,j]=0
plt.imsave('/home/russell/Pictures/DSP5/outfilePLT.jpg',corrMat,cmap='inferno')
tau=np.average(corrMat[np.nonzero(corrMat)])
tauArr=np.array(np.zeros((1,5)))
variance=0.0
for i in range(0,64):
    for j in range(0,64):
        variance=variance+(corrMat[i,j]-tau)**2
variance=variance/((64**2)-1)
variance=mt.sqrt(variance)
print(variance)
tauArr[0,0]=tau
tauArr[0,1]=tau+variance
tauArr[0,2]=tau+2*variance
tauArr[0,3]=tau+3*variance
tauArr[0,4]=tau+4*variance
print(tauArr)
corrTauMat=np.copy(corrMat)
for i in range(0,64):
    for j in range(0,64):
        if(corrTauMat[i,j]<tauArr[0,0]):
            corrTauMat[i,j]=0
plt.imsave('/home/russell/Pictures/DSP5/brainTauOne.jpg',corrTauMat,cmap='inferno')
corrTauMat=np.copy(corrMat)
for i in range(0,64):
    for j in range(0,64):
        if(corrTauMat[i,j]<tauArr[0,1]):
            corrTauMat[i,j]=0
plt.imsave('/home/russell/Pictures/DSP5/brainTauTwo.jpg',corrTauMat,cmap='inferno')
corrTauMat=np.copy(corrMat)
for i in range(0,64):
    for j in range(0,64):
        if(corrTauMat[i,j]<tauArr[0,2]):
            corrTauMat[i,j]=0
plt.imsave('/home/russell/Pictures/DSP5/brainTauThree.jpg',corrTauMat,cmap='inferno')
corrTauMat=np.copy(corrMat)
for i in range(0,64):
    for j in range(0,64):
        if(corrTauMat[i,j]<tauArr[0,3]):
            corrTauMat[i,j]=0
plt.imsave('/home/russell/Pictures/DSP5/brainTauFour.jpg',corrTauMat,cmap='inferno')
corrTauMat=np.copy(corrMat)
for i in range(0,64):
    for j in range(0,64):
        if(corrTauMat[i,j]<tauArr[0,4]):
            corrTauMat[i,j]=0
plt.imsave('/home/russell/Pictures/DSP5/brainTauFive.jpg',corrTauMat,cmap='inferno')
