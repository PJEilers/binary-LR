# encoding: utf-8
import numpy as np
import scipy.optimize as optimizer
import matplotlib.pyplot as plt
import time


class LogisticRegression:

    def __init__(self, trainDatapath):
        self.X, self.y = self.loadData(trainDatapath)

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def loadData(self, path):
        print "Loading data"
        X = []
        y = []
        fileIn = open(path)
        data = fileIn.read().split("\n")
        for row in data:
            inputTemp = row.split(" ")
            #in case there are more than one spaces
            inputData = [float(line) for line in inputTemp if line is not '']
            y.append(int(inputData[-1]))
            inputData.pop(-1)  # remove last column
            X.append([1.0] + inputData)  # add x0 as 1.0 to all rows
        fileIn.close()
        count = len(y)
        print "%d rows loaded into mainb.py" % (count)
        return np.matrix(X), np.matrix(y).T

    def plotInput(self, X, y):
        m, n = np.shape(X)
        if n != 3:
            print "Sorry! This only works for two variables"
            return 1
        print " Plotting the dataset ... "
        X = np.squeeze(np.asarray(X[:, 1:]))
        y = np.squeeze(np.asarray(y))

        ones = np.array([X[ind, :] for ind, val in enumerate(y) if val == 1])
        zeros = np.array([X[ind, :] for ind, val in enumerate(y) if val == 0])

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(ones[:, 0], ones[:, 1], s=50, c='b',
                   marker='x', label='ones')
        ax.scatter(zeros[:, 0], zeros[:, 1], s=50, c='r',
                   marker='o', label='zeros')
        ax.legend()
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')

    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))

    def costFunction(self, theta, X, y):
        m, n = X.shape
        theta = np.matrix(theta).T
        g_z = self.sigmoid(X * theta)
        H = np.multiply(-y, np.log(g_z))
        T = np.multiply((1 - y), np.log(1 - g_z))
        cost = (np.sum(H - T) / m)
        print H
        print T
        print cost
        print " -- --   H: T : cost - - - - -"
        return cost

    def gradient(self, theta, X, y):
        nData = X.shape[0]
        error = self.sigmoid(X * np.matrix(theta).T) - y
        grad = (1. / nData) * (X.T * error)
        return grad

    def predict(self, theta, X):
        response = self.sigmoid(X * theta.T)
        return [1 if x >= 0.5 else 0 for x in response]

    def plotDecisionBoundary(self, X, y, thetas):
        print ("new")
        X = np.squeeze(np.asarray(X[:, 1:]))
        y = np.squeeze(np.asarray(y))
        # evenly sampled points
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50))
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        #plot background colors
        ax = plt.gca()
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        #cs = ax.contourf(xx, yy, Z, cmap='RdBu', alpha=.5)
        cs2 = ax.contour(xx, yy, Z, cmap='RdBu', alpha=.5)
        plt.clabel(cs2, fmt='%2.1f', colors='k', fontsize=14)
        # Plot the points
        ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label='Negatif')
        ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label='Pozitif')
        # make legend
        plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
        plt.show()

    def train(self, X, y):
        print "training the dataset . . . "
        beginT = time.time()
        (trainX, trainY) = (X, y)
        (m, n) = trainX.shape
        thetas = np.zeros(n)
        trained = optimizer.fmin_tnc(func=self.costFunction,
                                     x0=thetas,
                                     fprime=self.gradient,
                                     args=(trainX, trainY),
                                     disp=False)
        theta_min = np.matrix(trained[0])
        print 'Training successfully completed \
                in %f seconds!' % (time.time() - beginT)
        return theta_min

    def test(self, X, y, theta_min):
        print "testing  . . . "
        testX = X
        testY = y
        nLines = len(testY)
        print 'Testing %d rows' % (nLines)
        predictions = self.predict(theta_min, testX)
        correct = [1 if ((p == 1 and t == 1) or (p == 0 and t == 0))
                   else 0 for (p, t) in zip(predictions, testY)]
        accuracy = sum(map(int, correct)) % len(correct)
        print " - "*10 + "Program Completed ! " + " - "*10
        print 'Correctly predicted = {0}% of '.format(accuracy) + '\
           %d test items ' % (nLines)
