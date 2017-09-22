"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
import sys

class RTLearner(object):

    def __init__(self,leaf_size=1,verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        #build tree for query later
        a = np.insert(dataX,dataX.shape[1],dataY,axis=1)
        self.tree = self.buildTree(np.insert(dataX,dataX.shape[1],dataY,axis=1))
        # print(dataX.shape[0])
        # print(self.tree)
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        y = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            y[i]=self.traverseTree(points[i,:])
        return y

    def buildTree(self, data):
        # leafnode = -2
        # no children = -1
        if(data.shape[0]<=self.leaf_size):
                return np.array([[-2,np.mean(data[:,data.shape[1]-1]),-1,-1]])

        if(all(data[:,data.shape[1]-1]==data[0,data.shape[1]-1])):
            return np.array([[-2,data[0,data.shape[1]-1],-1,-1]])
        else:
            split = True
            while(split):
                feature = np.random.randint(data.shape[1] - 1)
                splitVal = (data[np.random.randint(data.shape[0]), feature] + \
                            data[np.random.randint(data.shape[0]), feature]) / 2
                leftTree = data[data[:, feature] <= splitVal]
                rightTree = data[data[:, feature] > splitVal]
                if(leftTree.size==0 or rightTree.size==0):
                    continue
                else:
                    leftTree = self.buildTree(data[data[:, feature] <= splitVal])
                    rightTree = self.buildTree(data[data[:, feature] > splitVal])
                    split=False
            root = [feature,splitVal,1,leftTree.shape[0]+1]
            return np.vstack((root,leftTree,rightTree))

    def traverseTree(self,points):
        row =0
        value=1
        leaf =False
        while(not leaf):
            feature = self.tree[row][0]
            if(points[feature] <= self.tree[row][1]):
                row += self.tree[row][2]
            else:
                row += self.tree[row][3]
            leaf = (self.tree[row][0]==-2)
            value = self.tree[row][1]

        return value


if __name__=="__main__":
    print "the secret clue is 'zzyzx'"