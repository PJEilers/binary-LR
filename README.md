Binary Logistic Regression with Gradient Descent

Train and test methods can be called via command line with the following parameters:

    -i                        Absolute path to training data
    -o                        Absolute path to test data

Example : 

$ python mainb.py -i testData/test_data.txt -j testData/test_data.txt


mainb.py

   line: 
   
    trained = optimizer.fmin_tnc(func=costFunction,
                                 x0=thetas,
                                 fprime=gradientDescent,
                                 args=(trainX,trainY),
                                 disp=False)





