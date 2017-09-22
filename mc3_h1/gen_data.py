"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regresstion than random trees
def best4LinReg():
    X = np.random.randint(100,size=(100, 4))
    Y = 5 * X[:,0] + 2 * X[:,1] + 0.2 * X[:,2] - X[:,3]
    return X, Y

def best4RT():
    X = np.random.randint(10, size=(1000, 2))
    #X1= np.array([[1,2],[1,2],[8,8],[8,8],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8]])
    Y= []
    for row in X:
        if row[0]>8:
            if row[1]<5:
                Y.append(30)
            else:Y.append(-1)
        elif row[0] > 6:
            if row[1]<5:
                Y.append(30)
            else:Y.append(-1)
        elif row[0] > 6:
            if row[1]<5:
                Y.append(30)
            else:Y.append(-1)
        elif row[0] > 6:
            if row[1]<5:
                Y.append(30)
            else:Y.append(-1)
        else:
            if row[1]<5:
                Y.append(30)
            else:Y.append(-1)


    Y = np.array(Y)
    return X, Y

if __name__=="__main__":
    print "they call me Tim."
