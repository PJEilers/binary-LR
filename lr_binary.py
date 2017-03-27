# encoding: utf-8
import numpy as np
import scipy.optimize as optimizer
import matplotlib.pyplot as plt
import time

def loadData(path):
    print "Loading data"
    X = []
    y = []
    fileIn = open(path)
    data=fileIn.read().split("\n")
    for row in data:
        inputTemp = row.split(" ")
        #in case there are more than one spaces
        inputData=[float(line) for line in inputTemp if line is not '']        
        y.append(int(inputData[-1]))
        inputData.pop(-1)    
        #add x0 as 1.0 to all rows
        X.append([1.0]+inputData)
        
    fileIn.close()
    count = len(y)
    print "%d rows loaded into mainb.py" % (count)
    return np.matrix(X),np.matrix(y).T

    
def plotInput(X, y):
    m,n = np.shape(X)
    if n != 3:
        print "Sorry! This only works for two variables"
        return 1
    print " Plotting the dataset ... "    
    X = np.squeeze(np.asarray(X[:,1:]))
    y = np.squeeze(np.asarray(y))
    
    ones=np.array([X[ind,:] for ind,val in enumerate(y) if val == 1])
    zeros=np.array([X[ind,:] for ind,val in enumerate(y) if val == 0])
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(ones[:,0], ones[:,1], s=50, c='b',
               marker='x', label='ones')
    ax.scatter(zeros[:,0], zeros[:,1], s=50, c='r', 
               marker='o', label='zeros')
    ax.legend()
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
 

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def costFunction(theta, X, y):
    m,n = X.shape; 
    theta = np.matrix(theta).T
    H = np.multiply(-y, np.log(sigmoid(X * theta)))
    T = np.multiply((1 - y), np.log(1 - sigmoid(X * theta)))
    return np.sum(H - T) / m
    

def gradientDescent(theta, X, y):
    theta = np.matrix(theta).T
    nData = int(theta.ravel().shape[1])
    grad = np.zeros(nData)
    error = sigmoid(X * theta) - y
    
    for i in range(nData):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad

def predict(theta, X):
    response = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in response]

def plotDecisionBoundary(X, y,thetas):
    print "plot Decision Boundery"     
    

def train(path):
    print "training the dataset . . . "
    beginT = time.time()


    #main
    # initialize parameters and lables to matrixes 
    trainX,trainY=loadData(path)
    (m,n)=trainX.shape
    thetas=np.zeros(n)

   # plotInput(trainX,trainY)
   # print costFunction(thetas,trainX,trainY)
   # print gradientDescent(thetas,trainX,trainY)

    trained = optimizer.fmin_tnc(func=costFunction, x0=thetas, fprime=gradientDescent, args=(trainX,trainY), disp=False)
    theta_min = np.matrix(trained[0])
    print 'Training successfully completed in %f seconds!' % (time.time() - beginT)
    return theta_min

    
def test(path,theta_min):
    print "testing  . . . "
    testX,testY=loadData(path)
    nLines = len(testY)
    print 'Testing %d rows' % (nLines)
    predictions = predict(theta_min, testX)
    correct = [1 if ((p == 1 and t == 1) or (p == 0 and t == 0)) else 0 for (p, t) in zip(predictions, testY)]
    accuracy = sum(map(int, correct)) % len(correct)
    print " -   -    -    -    -    -   Program Completed !  -   -    -    -    -    -"
    print 'Correctly predicted = {0}% of '.format(accuracy) + '%d test items ' % (nLines)
    









