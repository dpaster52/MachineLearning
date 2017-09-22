"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import QLearner as ql
import pandas as pd
from util import get_data, plot_data
import numpy as np
import math

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.bins =5

    # this method should create a QLearner, and train it for trading
    # Daily return in percentage
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):
        syms = [symbol]
        #Call indicators function to receive indicators with dates before starting range
        normalAverage, bollBand, momentum, prices = self.plotIndicators(sd-dt.timedelta(days=10),ed,syms)
        #Get indicators for the given dates
        bollBand= bollBand.loc[prices.index>=sd]
        normalAverage = normalAverage.loc[prices.index >= sd]
        momentum = momentum.loc[prices.index >= sd]
        prices = prices.tail(len(momentum)+1)

        self.bins= int(math.floor(len(bollBand.index)/7)) if len(bollBand.index)>21 else 10

        #Discretize values in the dataframs
        bollBand = self.dfDiscretize(bollBand,steps=self.bins)
        normalAverage = self.dfDiscretize(normalAverage,steps=self.bins)
        momentum = self.dfDiscretize(momentum,steps=self.bins)

        #loop varialbles
        iterationMin = 3
        iterationMax =50
        firstState = (bollBand.iat[0,0],normalAverage.iat[0,0],momentum.iat[0,0])
        dims = (self.bins,self.bins,self.bins)
        firstState = np.ravel_multi_index(firstState, dims=dims)
        #List to hole end cash values
        returnArr = []
        #Initialize learner
        self.learner = ql.QLearner(num_states=self.bins**3,num_actions=5,alpha = 0.2, \
        gamma = 0.9, rar = 0.9, radr = 0.99, dyna = 0,verbose=False)

        for i in range(iterationMax):
            #Inner loop variables
            action = self.learner.querysetstate(firstState)
            shares = 0
            orders = pd.DataFrame(columns=["Cash", "Shares"], index=momentum.index)
            cash=sv
            for j in range(len(momentum.index)-1):
                #Buy 500 shares --0--
                if action==0 and shares in [0,-500]:
                    shares+=500
                    tOrder =500
                #Buy 1000 shares --1--
                elif action == 1 and shares==-500:
                    shares+=1000
                    tOrder = 1000
                #Sell 500 shares --2--
                elif action == 2 and shares in [0, 500]:
                    shares-=500
                    tOrder = -500
                #Sell 1000 shares --3--
                elif action == 3 and shares==500:
                    shares-=1000
                    tOrder = -1000
                # Do Nothing --4--
                else:
                    tOrder=0
                stockMoney = tOrder*prices.iat[j+1,0]
                daily= (prices.iat[j+1,0]/prices.iloc[j,0])-1
                cash -=stockMoney
                reward = daily*shares
                #Debug Information
                orders.iloc[j]=pd.Series({"Cash":cash,"Shares":shares})
                #Update learner
                nextState = np.ravel_multi_index((bollBand.iat[j,0],normalAverage.iat[j,0],momentum.iat[j,0]), dims=dims)
                action = self.learner.query(nextState,reward)
            j+=1
            if shares!=0:
                shares = 500 if shares<0 else -500
                stockMoney = shares * prices.iat[j + 1,0]
                cash -= stockMoney
                orders.iloc[j] = pd.Series({"Cash": cash, "Shares": 0})
            else:
                orders.iloc[j] = pd.Series({"Cash": cash, "Shares": 0})

            #print cash
            returnArr.append(cash)
            #print "Reture val every: ",returnArr[i]
            if i > iterationMin-1:
                if returnArr[i] == returnArr[i-2]:
                    #print "Converged ",i
                    #print "Return val",returnArr[i]
                    break





    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        # Call indicators function to receive indicators with dates before starting range
        normalAverage, bollBand, momentum, prices = self.plotIndicators(sd - dt.timedelta(days=20), ed, syms)

        # Get indicators for the given dates
        bollBand = bollBand.loc[prices.index >= sd]
        normalAverage = normalAverage.loc[prices.index >= sd]
        #MACDInd = MACDInd.loc[prices.index >= sd]
        momentum = momentum.loc[prices.index >= sd]
        prices = prices.loc[prices.index >= sd]

        # Discretize values in the dataframs

        bollBand = self.dfDiscretize(bollBand, steps=self.bins)
        normalAverage = self.dfDiscretize(normalAverage, steps=self.bins)
        #MACDInd = self.dfDiscretize(MACDInd, steps=self.bins)
        momentum = self.dfDiscretize(momentum, steps=self.bins)
        firstState = (bollBand.iat[0,0], normalAverage.iat[0,0], momentum.iat[0,0])
        dims = (self.bins, self.bins, self.bins)
        firstState = np.ravel_multi_index(firstState, dims=dims)
        self.learner = ql.QLearner(num_states=self.bins ** 3, num_actions=5,rar=0.98, verbose=False)
        action = self.learner.querysetstate(firstState)
        shares = 0
        orders = pd.DataFrame(columns=["Orders"], index=prices.index)
        for j in range(len(prices.index)):
            # Buy 500 shares --0--
            if action == 0 and shares in [0, -500]:
                shares += 500
                orders.iat[j,0]=500
            #Buy 1000 shares --1--
            elif action == 1 and shares == -500:
                shares += 1000
                orders.iat[j,0] = 1000
            # Sell 500 shares --2--
            elif action == 2 and shares in [0, 500]:
                shares -= 500
                orders.iat[j,0] = -500
            # Sell 1000 shares --3--
            elif action == 3 and shares == 500:
                shares -= 1000
                orders.iat[j,0] = -1000
            # Do Nothing --4--
            else:
                orders.iat[j,0] = 0
            nextState = np.ravel_multi_index((bollBand.iat[j,0], normalAverage.iat[j,0],\
                    momentum.iat[j,0]), dims=dims)

            action = self.learner.querysetstate(nextState)

        return orders
    def dfDiscretize(self, df, steps=10):

        threshold = []
        stepSize = df.shape[0]/steps
        sym = df.columns.values[0]
        dfSort = df.sort_values(sym)

        for i in range(0,steps-1):
            threshold.append(dfSort[sym].iloc[(i+1) * stepSize])

        df[sym] = np.digitize(df[sym].values, threshold)
        return df

    def plotIndicators(self,sd = dt.datetime(2006,1,1), ed = dt.datetime(2009,12,31), \
        syms = ['IBM'],gen_plot=False):

        # Read in adjusted closing prices for given symbols, date range
        dates = pd.date_range(sd, ed)
        prices_all = get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        # Normalize prices according to first day
        normPrice = prices / prices.iloc[0]-1

        
        # Momentum -0.5 - 0.5
        tradingDays = 5
        momentum = (prices / prices.shift(tradingDays)) - 1

        # Simple Moving Average -0.5 - 0.5
        movingAverage = pd.rolling_mean(prices,tradingDays)

        #Prices / SMA
        normalAverage = prices/movingAverage-1

        # Bollinger Bands  <-1 buy >1 sell
        std = pd.rolling_std(prices,tradingDays)
        bollBand = (prices-movingAverage)/(2*std)

        return normalAverage,bollBand,momentum,prices


if __name__=="__main__":
    print "One does not simply think up a strategy"
