#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import getopt
import lr_binary_class as lr


def main(argv):
    #trainFile = ''
    #testFile = ''
    trainFile = 'testData/test_data.txt'
    testFile = trainFile
    """
    usage()
    try:
        opts, args = getopt.getopt(
            argv, "hi:j:", ["trainingFile=", "testFile="])
    except getopt.GetoptError:
        # print usage
        print 'mainb.py -i <trainingFilet> -j <testFile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'mainb.py -i <trainingFilet> -j <testFile>'
            sys.exit()
        elif opt in ("-i", "--trainingFile"):
            trainFile = arg
        elif opt in ("-j", "--testFile"):
            testFile = arg
    """
    print "   step-1 : "

    lr_c = lr.LogisticRegression(trainFile)
    train_X = lr_c.get_X()
    train_y = lr_c.get_y()
    #lr_c.plotInput(train_X, train_y)
    weights = lr_c.train(train_X, train_y)

    print 'test data is: ', testFile
    print "   step-2 : "

    test_X = lr_c.get_X()
    test_y = lr_c.get_y()

    lr_c.test(test_X, test_y, weights)
    #lr_c.plotDecisionBoundary(train_X, train_y, weights)


def usage():
    print "usage: "
    usage = """
    -h --help                 Prints this
    -i                        Absolute path to training data
    -j                        Absolute path to test data
    """
    print usage


if __name__ == "__main__":
    main(sys.argv[1:])
