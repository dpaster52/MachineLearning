"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
from datetime import timedelta
import QLearner as ql

import pandas as pd
import pandas as pd
import numpy as np
import util as ut

from sys import exit


class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), sv = 10000): 
        prices,momentum, BB, SMA, EOM = self.computeIndicators(symbol,sd,ed)
        dates = momentum.index

        # Discretize all indicators
        numBins = 3
        totalBins = numBins + 1
        momentum,BB,SMA,EOM = self.discretize([momentum,BB,SMA,EOM],numBins)
                
        # Initialize holdings and the number of holdings states (possible positons) 
        holdings = 0
        numHoldStates = 3 # 0/-500/500 == 0/1/2
        
        # Holdings constants
        ZERO = 0
        MINUS500 = 1
        PLUS500 = 2
        
        # Compute initial state
        inData = (holdings,momentum.iloc[0][symbol],BB.iloc[0][symbol],SMA.iloc[0][symbol],EOM.iloc[0][symbol])
        stateDimensions = (numHoldStates,totalBins,totalBins,totalBins,totalBins)
        initState = self.computeState(inData,stateDimensions)
        
        # Instantiate a Q-learner
        numStates = numHoldStates*((totalBins) ** 4)
        numActions = 3 # 0 = doNothing/ 1 = short/ 2 = long
        self.learner = ql.QLearner(num_states=numStates,num_actions=numActions, alpha = 0.2, gamma = 0.9, rar = 0.98, radr = 0.999, dyna = 0, verbose=False)        

        # Loop variables
        dfTrades = pd.DataFrame(columns=['Cash','Holdings','Reward'],index=dates)
        dfTrades.ix[0]['Cash'] = sv
        converged = False
        iters = 0
        minIters = 10
        maxIters = 75
        posReward = 700
        negReward = -500
        cumRewards = []

        while not(converged and iters > minIters) and (iters < maxIters): 
            action = self.learner.querysetstate(initState) # 0 = doNothing/ 1 = short/ 2 = long
            holdings = 0

            for date in dates:
                # Compute yesterdays return
                yDayPrice = prices.ix[prices.index.get_loc(date) - 1][symbol] 
                yRet = holdings * yDayPrice
               
                canGoShort = holdings in [0,500]
                canGoLong = holdings in [0,-500]

                # Compute new holdings and update state
                if action == 1 and canGoShort:
                    # Go short
                    #print "short:", holdings
                    holdings += -500
                elif action == 2 and canGoLong:
                    # Go long
                    #print "long:", holdings
                    holdings += 500

                # Calculate todays return and daily return
                tRet = holdings * prices.loc[date][symbol]
                dailyRet = (tRet/ yRet) - 1

                if date == dates[0]:
                    cash = sv 
                else:
                    cash = dfTrades.ix[dfTrades.index.get_loc(date) - 1]['Cash'] + tRet

                # Update trades
                reward = posReward if dailyRet > 0 else negReward
                dfTrades.loc[date]= pd.Series({'Cash':cash, 'Holdings':holdings, 'Reward':reward})

                # Compute next state
                discreteHolding = ZERO if holdings==0 else (PLUS500 if holdings==500 else MINUS500)
                inData = (discreteHolding,momentum.loc[date][symbol],BB.loc[date][symbol],SMA.loc[date][symbol],EOM.loc[date][symbol])
                #stateDimensions = (numHoldStates,totalBins,totalBins,totalBins,totalBins)
                nextState = self.computeState(inData,stateDimensions)
                
                # Get next action
                action = self.learner.query(nextState, reward)

            #print dfTrades
            cumRewards.append(dfTrades.loc[dates[-1]]['Cash'])

            # Determine convergence
            if iters > 1:
                currCumReward = cumRewards[iters]
                convergeFactor = 0.05 * cumRewards[iters-1]
                if (currCumReward - convergeFactor) < cumRewards[iters-1] < (currCumReward + convergeFactor):
                    # Converge if cumRet[this iteration] approx equal to cumRet[last iteration]
                    converged = True

            iters += 1


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", sd=dt.datetime(2009,1,1), ed=dt.datetime(2010,1,1), sv = 10000):
        prices,momentum, BB, SMA, EOM = self.computeIndicators(symbol,sd,ed)
        dates = momentum.index
        
        # Discretize all indicators
        numBins = 3
        totalBins = numBins + 1
        momentum,BB,SMA,EOM = self.discretize([momentum,BB,SMA,EOM],numBins)

        # Initialize holdings and the number of holdings states (possible positons) 
        holdings = 0
        numHoldStates = 3 # 0/-500/500 == 0/1/2
        
        # Holdings constants
        ZERO = 0
        MINUS500 = 1
        PLUS500 = 2
        
        # Compute initial state
        inData = (holdings,momentum.iloc[0][symbol],BB.iloc[0][symbol],SMA.iloc[0][symbol],EOM.iloc[0][symbol])
        stateDimensions = (numHoldStates,totalBins,totalBins,totalBins,totalBins)
        initState = self.computeState(inData,stateDimensions)
        
        # Loop variables
        action = self.learner.querysetstate(initState) # 0 = doNothing/ 1 = short/ 2 = long
        dfOrders = pd.DataFrame(columns=['Orders'],index=dates)

        for date in dates:
            #  Skip to allow for balancing on the last day
            #if date == dates[-1]: break

            canGoShort = holdings in [0,500]
            canGoLong = holdings in [0,-500]

            # Compute new holdings and update state
            if action == 1 and canGoShort:
                # Go short
                #print "short {} {}".format(holdings,date)
                holdings += -500
                dfOrders.loc[date]= pd.Series({'Orders':-500})
            elif action == 2 and canGoLong:
                # Go long
                #print "long {} {}".format(holdings,date)
                holdings += 500
                dfOrders.loc[date]= pd.Series({'Orders':500})
            else:
                dfOrders.loc[date]= pd.Series({'Orders':0})

            # Compute next state
            discreteHolding = ZERO if holdings==0 else (PLUS500 if holdings==500 else MINUS500)
            inData = (discreteHolding,momentum.loc[date][symbol],BB.loc[date][symbol],SMA.loc[date][symbol],EOM.loc[date][symbol])
            nextState = self.computeState(inData,stateDimensions)

            # Get next action
            action = self.learner.querysetstate(nextState)
        '''
        if dfOrders.Holdings.sum() != 0:
            holdings = -500 if dfOrders.Holdings.sum() == 500 else 500
            dfOrders.loc[dates[-1]]= pd.Series({'Orders':holdings})
        '''

        #print dfOrders
        return dfOrders


    def computeState(self,inData,stateDimensions):
        state = np.ravel_multi_index(inData, dims=stateDimensions)
        return state


    def discretize(self, dfs, steps):
        outDFs = []

        for df in dfs:
            threshold = [0] * steps
            stepSize = df.shape[0]/steps
            sym = df.columns.values[0]
            sortedDF = df.sort_values(sym)

            for i in range(0,steps):
                threshold[i] = sortedDF[sym].iloc[(i+1) * stepSize]

            df[sym] = np.digitize(df[sym].values, threshold)
            outDFs.append(df)

        return outDFs[0],outDFs[1],outDFs[2],outDFs[3]


    def computeIndicators(self, symbol, sd, ed):
        syms=[symbol]
        lookAhead = 10

        # Read in prices, highs, lows, and volume        
        dates = pd.date_range(sd-timedelta(days=lookAhead), ed)
        prices_all = ut.get_data(syms, dates)  
        highs_all = ut.get_data(syms, dates, colname='High')  
        lows_all = ut.get_data(syms, dates, colname='Low')  
        volume_all = ut.get_data(syms, dates, colname = "Volume")  

        prices = prices_all[syms]  
        highs = highs_all[syms]
        lows = lows_all[syms]
        volumes = volume_all[syms]  

        if self.verbose: 
            print prices 
            print volume
  
        # Compute indicators
        # indicator 1: Momentum --> [-0.5,+0.5]
        winSize = 3
        momentum = (prices / prices.shift(winSize)) - 1

        # indicator 2: Simple Moving Average --> [-0.5,+0.5]
        rollingMean = pd.rolling_mean(prices,winSize)
        SMA = (prices / rollingMean) - 1 

        # indicator 3: Bollinger Bands --> [-1,+1]
        stdDev = pd.rolling_std(prices,winSize)
        BB = (prices - rollingMean) / (2 * stdDev)
        upper = rollingMean + 2*stdDev
        lower = rollingMean - 2*stdDev

        # indicator 4: Ease of Movement
        distanceMoved = ((highs + lows) / 2) - ((highs.shift(1) + lows.shift(1))/ 2)
        boxRatio = (volumes/100000000) / (highs - lows) 
        EOM = distanceMoved / boxRatio
        EOM = pd.rolling_mean(EOM, window=winSize)

        # Re-index indicators to be in correct date frame
        momentum = momentum.iloc[momentum.index >= sd]
        BB = BB.iloc[BB.index >= sd]
        SMA = SMA.iloc[SMA.index >= sd]
        EOM = EOM.iloc[EOM.index >= sd]

        return prices,momentum,BB,SMA,EOM


if __name__=="__main__":
    print "One does not simply think up a strategy"
