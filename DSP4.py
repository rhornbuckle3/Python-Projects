import numpy as np 
import math as mt  
import collections as col
import cv2
#import pillow as pl     
book=open("austen-pride-757.txt","r")
image=open("place_final.jpg","r")
c=col.Counter(book.read())
bookSort=list(c.elements())
bookLen=len(bookSort)
#print(bookLen)
bookTab=set(bookSort)
bookTab=list(bookTab)
#print(bookTab)
bookCount=bookTab.copy()
for i in range(0,len(bookCount)):
    bookCount[i]=bookSort.count(bookTab[i])
#print(bookCount)
bookEntropy=np.array(np.zeros((len(bookCount))))
print(bookEntropy.size)
for i in range(0,bookEntropy.size):
    bookEntropy[i]=(bookCount[i]/bookLen)*mt.log2(bookLen/bookCount[i])
fullBookEntropy=bookEntropy.sum()*bookLen
print(fullBookEntropy)
#book is done


#phototime
imagr=cv2.imread("place_final.jpg")
imagrARR=np.array(list(imagr))
#print(imagrARR.shape)
imgCOLREF=np.array(np.zeros((256,256,256)))
for i in range(0,imagrARR.shape[0]):
    for j in range(0,imagrARR.shape[1]):
        imgCOLREF[imagrARR[i,j,0],imagrARR[i,j,1],imagrARR[i,j,2]]=imgCOLREF[imagrARR[i,j,0],imagrARR[i,j,1],imagrARR[i,j,2]]+1
imgCOLFIN=imgCOLREF[np.nonzero(imgCOLREF)]
imgEntropy=np.array(np.zeros(imgCOLFIN.shape))
for i in range(0,imgEntropy.size):
    imgEntropy[i]=(imgCOLFIN[i]/imagrARR.size)*mt.log2(imagrARR.size/imgCOLFIN[i])
fullImageEntropy=imgEntropy.sum()*imagrARR.size/3
print(fullImageEntropy)