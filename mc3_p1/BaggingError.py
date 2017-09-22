"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import RTLearner as rt
import BagLearner as bl
import sys

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = math.floor(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print testX.shape
    print testY.shape
    # create a learner and train it
    length = 75
    rmseIn=[]
    rmseOut=[]
    index = [i for i in range(1,length+1)]

    for i in range(length):
        for j in range(2):
            avgIn = []
            avgOut = []
            learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": length+1}, bags=10, boost=False, verbose=False)
            learner.addEvidence(trainX, trainY) # train it

            # evaluate in sample
            predY = learner.query(trainX) # get the predictions
            #avgIn.append(np.corrcoef(predY, y=trainY)[0,1])
            avgIn.append(math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0]))
            # print(math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0]))
            # evaluate out of sample
            predY = learner.query(testX) # get the predictions
            avgOut.append(math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0]))
            # print(math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0]))
            #avgOut.append(np.corrcoef(predY, y=testY)[0,1])
        rmseIn.append(np.mean(avgIn))
        rmseOut.append(np.mean(avgOut))
    df_temp = pd.concat([pd.DataFrame(rmseIn,index=index),pd.DataFrame(rmseOut,index=index)],keys=["In Sample","Out of Sample"],axis=1)
    graph = df_temp.plot(title="Bagging=10 ")
    graph.set_xlabel("Leaf size")
    graph.set_ylabel("Error")
    plt.grid()
    plt.show()
