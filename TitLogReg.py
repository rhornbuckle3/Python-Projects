import numpy as np
import pandas as pd
import math as mt
import numpy.random 

titanic_data=pd.read_csv('titanic_data.csv')
#get the data into separate training and testing sets
titanic_data=titanic_data.T
titanic_data=titanic_data.values
beta=None
oBeta=None
yData=None
xData=None
yTest=None
xTest=None
#Functions
def shuffleDataPrep():
    np.random.shuffle(np.transpose(titanic_data))
    #creating an 80-20 split in training data for testing
    trainSplit=mt.floor(titanic_data.shape[1]*.8)
    global xData
    xData=titanic_data[1:8,0:trainSplit]
    global xTest
    xTest=titanic_data[1:8,trainSplit:titanic_data.shape[1]+1]
    #gotta get our y Vectors into proper form (python has some hoops to jump through with 2d arrays of 1 row)
    global yData
    yData=titanic_data[0,0:trainSplit]
    yData=yData.reshape((1,-1))
    global yTest
    yTest=titanic_data[0,trainSplit:titanic_data.shape[1]+1]
    yTest=yTest.reshape((1,-1))
    global beta
    #The following are various weight vectors that I've tried and the % error(s) that they achieved
    beta=np.array([[-0.13022121],[ 1.13292532],[-0.02734318],[ 0.01320409],[ 0.53387497],[ 0.00646553]])
    #^ My final initial weight vector 
    #beta=np.array([[ 0.33483785],[ 0.81135626],[-0.80699351],[ 0.10440245],[ 0.53536229],[ 0.01398352]])
    #beta=np.array([[-1*abs(np.random.normal(-0.13694547,1e-2))],[abs(np.random.normal(0.90101702,1e-1))],[-1*abs(np.random.normal(-0.02776487,1e-2))],[abs(np.random.normal(0.01565402,1e-3))],[abs(np.random.normal(0.52777695,1e-2))],[abs(np.random.normal(0.00810008,1e-5))]])
    #beta=np.array([[-1*abs(np.random.normal(-0.14577136,1e-2))],[abs(np.random.normal(0.95797346,1e-1))],[-1*abs(np.random.normal(-0.02756235,1e-3))],[abs(np.random.normal(0.0144251,1e-3))],[abs(np.random.normal(0.52332493,1e-2))],[abs(np.random.normal(0.00802794,1e-4))]])
    #beta=np.array([[-0.19316827],[ 0.9737353 ],[-0.02428369],[-0.02648061],[ 0.49126379],[ 0.00808396]])
    #26,26,30,26,27,25,28,25,26,25
    #beta=np.array([[-0.14577136],[ 0.95797346],[-0.02756235],[ 0.0144251 ],[ 0.52332493],[ 0.00802794]])
    #29,24,23,28,26,23,29
    #beta=np.array([[-0.07531058],[ 0.94639399],[-0.03435913],[ 0.07294685],[ 0.56011063],[ 0.00695853]])
    #27,23
    #beta=np.array([[ 0.00214403],[ 0.93081158],[-0.03829632],[ 0.13714383],[ 0.60051621],[ 0.00560207]])
    #28,37,34,28
    #beta=np.array([[ 0.10257939],[ 0.9170227 ],[-0.04290944],[ 0.20882552],[ 0.64937154],[ 0.00515416]])
    #29,25
    #beta=np.array([[-0.1369454 ],[ 0.90101657],[-0.02776486],[ 0.01565401],[ 0.52777669],[ 0.00810007]])
    #beta=np.random.uniform(size=xData.shape[0],)
    #beta=np.random.normal(0,.5,size=xData.shape[0],)
    global oBeta
    beta=beta.reshape((-1,1))
    oBeta=np.array(beta)
def recFunctONE(i,iBeta,xSet):
    #this is the actual classifier
    xTemp=xSet[:,i].reshape((-1,1))
    return 1/(1+mt.e**np.matmul(np.transpose(-iBeta),xTemp))
def classifier(iBeta,xSet,ySet):
    #this is the full form of function 2.2
    lprob=0
    for i in range(0,xSet.shape[1]-1):
        lprob+=np.log((recFunctONE(i,iBeta,xSet)**np.asscalar(ySet[:,i]))*((1-recFunctONE(i,iBeta,xSet))**(1-np.asscalar(ySet[:,i]))))
    return lprob
def gradientProb(iBeta,xSet,ySet):
    gradLProb=0
    for i in range(0,xSet.shape[1]-1):
        gradLProb+=np.multiply(ySet[:,i]-recFunctONE(i,iBeta,xSet),xSet[:,i].reshape((-1,1)))
    return gradLProb
def gradientAscent(eta,tol,iBeta,xSet,ySet):
    while(True):
        grad=np.array(iBeta)
        print (grad)
        np.add(iBeta,eta*gradientProb(iBeta,xSet,ySet),out=iBeta)
        if((np.argmax(abs(iBeta-grad)))<tol):
            return iBeta
def error(iBeta,xSet,ySet):
    itera=0
    for i in range(0,xSet.shape[1]-1):
        if(round(np.asscalar(recFunctONE(i,iBeta,xSet)))!=ySet[:,i]):
            itera=itera+1
    return itera
while (True):
    #The infinite loop here is for finding viable weight vectors, I comment out the break at the end if I want to leave it going
    shuffleDataPrep()
    beta=gradientAscent(1e-7,1e-9,beta,xData,yData)
    print(classifier(beta,xTest,yTest))
    f=open("LogRegStatsRan.txt","a+")
    f.write('\n')
    f.write(str(100*(error(beta,xTest,yTest)/yTest.shape[1])))
    f.write('\n')
    f.write(np.array_str(oBeta))
    f.write('\n')
    f.write(np.array_str(beta))
    print(100*(error(beta,xTest,yTest)/yTest.shape[1]),"% Error Rate")
    print(oBeta)
    print(beta)
    f.close()
    print(recFunctONE(0,beta,np.array([[2],[0],[22],[0],[1],[20.66]])))
    #break