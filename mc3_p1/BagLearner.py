"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
import RTLearner as rt
import math


class BagLearner(object):
    def __init__(self, learner=rt.RTLearner,kwargs={"leaf_size":1},bags=20,boost=False,verbose=False):
        self.bags = bags
        self.boost=boost
        self.verbose =verbose
        self.learners = []

        for i in range(self.bags):
            self.learners.append(learner(**kwargs))



    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        for i in range(self.bags):
            newIndex = np.random.randint(0,high=dataX.shape[0],size=math.floor(dataX.shape[0]*.75))
            self.learners[i].addEvidence(dataX[newIndex],dataY[newIndex])

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        results = []

        for i in range(self.bags):
            results.append(self.learners[i].query(points))

        #print len(results[0])
        return np.mean(results,axis=0)


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
