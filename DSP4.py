import numpy as np 
import math as mt  
import collections as col
import cv2
book=open("austen-pride-757.txt","r")
c=col.Counter(book.read())
bookSort=list(c.elements())
bookLen=len(bookSort)
bookTab=set(bookSort)
#^Lets me get a list of unique occurences
bookTab=list(bookTab)
bookCount=bookTab.copy()
for i in range(0,len(bookCount)):
    bookCount[i]=bookSort.count(bookTab[i])
    #^getting the count for each character
bookEntropy=np.array(np.zeros((len(bookCount))))
for i in range(0,bookEntropy.size):
    #calculating the actual entropy
    bookEntropy[i]=(bookCount[i]/bookLen)*mt.log2(bookLen/bookCount[i])
print(bookEntropy.sum())
fullBookEntropy=bookEntropy.sum()*bookLen
print(fullBookEntropy)
#book is done
#phototime
imagrARR=np.array(list(cv2.imread("place_final.jpg")))
#getting the image into an array so that I can work on it
imgCOLREF=np.array(np.zeros((256,256,256)))
#I'll fit an array to the 256^3 color index and use that to keep track of the # of occurences of each color. It's a bit brute force-eske
#this is messy, I should do something smarter
for i in range(0,imagrARR.shape[0]):
    for j in range(0,imagrARR.shape[1]):
        imgCOLREF[imagrARR[i,j,0],imagrARR[i,j,1],imagrARR[i,j,2]]=imgCOLREF[imagrARR[i,j,0],imagrARR[i,j,1],imagrARR[i,j,2]]+1
imgCOLFIN=imgCOLREF[np.nonzero(imgCOLREF)]
#getting all of the non-zero occurences into a separate array
imgEntropy=np.array(np.zeros(imgCOLFIN.shape))
for i in range(0,imgEntropy.size):
    #calculating the entropy
    imgEntropy[i]=(imgCOLFIN[i]/imagrARR.size)*mt.log2(imagrARR.size/imgCOLFIN[i])
print(imgEntropy.sum())
fullImageEntropy=imgEntropy.sum()*imagrARR.size/3
print(fullImageEntropy)