"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut
from indicators import plotIndicators
import numpy as np

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
        dates = pd.date_range(sd, ed)
        #Call indicators function to receive indicators with dates before starting range
        normalAverage, bollBand, MACDInd, momentum, prices = plotIndicators(sd-dt.timedelta(days=80),ed,syms)

        #Get indicators for the given dates
        bollBand= bollBand.loc[prices.index>=sd]
        normalAverage = normalAverage.loc[prices.index >= sd]
        MACDInd = MACDInd.loc[prices.index >= sd]
        momentum = momentum.loc[prices.index >= sd]
        prices = prices.loc[prices.index >= sd]

        #Discretize values in the dataframs
        bins =self.bins
        bollBand = self.dfDiscretize(bollBand,steps=self.bins)
        normalAverage = self.dfDiscretize(normalAverage,steps=self.bins)
        MACDInd = self.dfDiscretize(MACDInd,steps=self.bins)
        momentum = self.dfDiscretize(momentum,steps=self.bins)

        #loop varialbles
        iterationMin = 3
        iterationMax =50
        rewardPos = 100
        rewardNeg =-2000
        firstState = (bollBand.iloc[0][symbol],normalAverage.iloc[0][symbol],MACDInd.iloc[0][symbol],momentum.iloc[0][symbol])
        dims = (self.bins,self.bins,self.bins,self.bins)
        firstState = np.ravel_multi_index(firstState, dims=dims)
        returnArr = []
        self.learner = ql.QLearner(num_states=self.bins**4,num_actions=5,verbose=False)
        orders = pd.DataFrame(columns=["Cash","Shares"],index=prices.index)

        for i in range(iterationMax):
            action = self.learner.querysetstate(firstState)
            shares = 0
            for j in range(len(prices.index)):
                lastRet = shares * prices.iloc[j-1][symbol]
                #Buy allowed
                if action==0 and shares in [0,-500]:
                    shares+=500
                elif action == 1 and shares==-500:
                    shares+=1000
                #sell allowed
                elif action == 2 and shares in [0, 500]:
                    shares-=500
                elif action == 3 and shares==500:
                    shares-=1000
                ret = shares*prices.iloc[j][symbol]
                daily = ret/lastRet-1
                if j==0:
                    cash = sv
                else:
                    cash = orders.iloc[j-1]["Cash"]+ret
                reward = rewardPos if daily>0 else rewardNeg

                orders.iloc[j]=pd.Series({"Cash":cash,"Shares":shares})
                nextState = np.ravel_multi_index((bollBand.iloc[j][symbol],normalAverage.iloc[j][symbol],MACDInd.iloc[j][symbol],momentum.iloc[j][symbol]), dims=dims)

                action = self.learner.query(nextState,reward)
            returnArr.append(cash)
            if i>iterationMin-1:
                converge = 0.1 * returnArr[i - 1]
                if (returnArr[i] - converge) < returnArr[i-1] < (returnArr[i] + converge):
                    break





    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        # Call indicators function to receive indicators with dates before starting range
        normalAverage, bollBand, MACDInd, momentum, prices = plotIndicators(sd - dt.timedelta(days=80), ed, syms)

        # Get indicators for the given dates
        bollBand = bollBand.loc[prices.index >= sd]
        normalAverage = normalAverage.loc[prices.index >= sd]
        MACDInd = MACDInd.loc[prices.index >= sd]
        momentum = momentum.loc[prices.index >= sd]
        prices = prices.loc[prices.index >= sd]

        # Discretize values in the dataframs

        bollBand = self.dfDiscretize(bollBand, steps=self.bins)
        normalAverage = self.dfDiscretize(normalAverage, steps=self.bins)
        MACDInd = self.dfDiscretize(MACDInd, steps=self.bins)
        momentum = self.dfDiscretize(momentum, steps=self.bins)
        firstState = (bollBand.iloc[0][symbol], normalAverage.iloc[0][symbol], MACDInd.iloc[0][symbol], momentum.iloc[0][symbol])
        dims = (self.bins, self.bins, self.bins, self.bins)
        firstState = np.ravel_multi_index(firstState, dims=dims)
        self.learner = ql.QLearner(num_states=self.bins ** 4, num_actions=5,rar=0.98, verbose=False)

        action = self.learner.querysetstate(firstState)
        shares = 0
        orders = pd.DataFrame(columns=["Orders"], index=prices.index)
        for j in range(len(prices.index)):
            # Buy allowed
            if action == 0 and shares in [0, -500]:
                shares += 500
                orders.iloc[j]["Orders"]=500
            elif action == 1 and shares == -500:
                shares += 1000
                orders.iloc[j]["Orders"] = 1000
            # sell allowed
            elif action == 2 and shares in [0, 500]:
                shares -= 500
                orders.iloc[j]["Orders"] = -500
            elif action == 3 and shares == 500:
                shares -= 1000
                orders.iloc[j]["Orders"] = -1000
            else:
                orders.iloc[j]["Orders"] = 0
            nextState = np.ravel_multi_index((bollBand.iloc[j][symbol], normalAverage.iloc[j][symbol],
                                              MACDInd.iloc[j][symbol], momentum.iloc[j][symbol]), dims=dims)

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

if __name__=="__main__":
    print "One does not simply think up a strategy"
'''
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
        dates = pd.date_range(sd, ed)
        #Call indicators function to receive indicators with dates before starting range
        normalAverage, bollBand, MACDInd, momentum, prices = self.plotIndicators(sd-dt.timedelta(days=56),ed,syms)
        #Get indicators for the given dates
        bollBand= bollBand.loc[prices.index>=sd]
        normalAverage = normalAverage.loc[prices.index >= sd]
        MACDInd = MACDInd.loc[prices.index >= sd]
        momentum = momentum.loc[prices.index >= sd]
        prices = prices.tail(len(momentum)+1)
        self.bins= int(math.floor(len(bollBand.index)/7)) if len(bollBand.index)>21 else 3

        #Discretize values in the dataframs
        bollBand = self.dfDiscretize(bollBand,steps=self.bins)
        normalAverage = self.dfDiscretize(normalAverage,steps=self.bins)
        MACDInd = self.dfDiscretize(MACDInd,steps=self.bins)
        momentum = self.dfDiscretize(momentum,steps=self.bins)

        #loop varialbles
        iterationMin = 3
        iterationMax =50
        rewardPos = 100
        rewardNeg =-2000
        firstState = (bollBand.iloc[0][symbol],normalAverage.iloc[0][symbol],MACDInd.iloc[0][symbol],momentum.iloc[0][symbol])
        dims = (self.bins,self.bins,self.bins,self.bins)
        firstState = np.ravel_multi_index(firstState, dims=dims)
        returnArr = []
        #TODO 3 actions
        self.learner = ql.QLearner(num_states=self.bins**4,num_actions=5,alpha = 0.2, \
        gamma = 0.9, rar = 0.9, radr = 0.99, dyna = 0,verbose=False)
        #orders = pd.DataFrame(columns=["Cash","Shares"],index=prices.index)

        for i in range(iterationMax):
            action = self.learner.querysetstate(firstState)
            shares = 0
            orders = pd.DataFrame(columns=["Cash", "Shares"], index=momentum.index)
            cash=sv
            lastshares=0
            for j in range(len(momentum.index)-1):
                #Buy allowed
                if action==0 and shares in [0,-500]:
                    shares+=500
                    tOrder =500
                elif action == 1 and shares==-500:
                    shares+=1000
                    totOrder = 1000
                #sell allowed
                elif action == 2 and shares in [0, 500]:
                    shares-=500
                    tOrder = -500
                elif action == 3 and shares==500:
                    shares-=1000
                    tOrder = -1000
                else:
                    tOrder=0

                stockMoney = tOrder*prices.iloc[j+1][symbol]
                daily= (prices.iloc[j+1][symbol]/prices.iloc[j][symbol])-1


                cash -=stockMoney
                reward = rewardPos if daily>0 else rewardNeg
                # if j == len(momentum.index)-1:
                #     if shares!=0:
                #         ret = abs(shares) * prices.iloc[j + 1][symbol]
                #         cash = orders.iloc[j - 1]["Cash"] + ret

                orders.iloc[j]=pd.Series({"Cash":cash,"Shares":shares})
                nextState = np.ravel_multi_index((bollBand.iloc[j][symbol],normalAverage.iloc[j][symbol],MACDInd.iloc[j][symbol],momentum.iloc[j][symbol]), dims=dims)
                action = self.learner.query(nextState,reward)
                lastshares=orders.iloc[j]["Shares"]
            j+=1
            if shares!=0:
                shares = 500 if shares<0 else -500
                stockMoney = shares * prices.iloc[j + 1][symbol]
                cash -= stockMoney
                orders.iloc[j] = pd.Series({"Cash": cash, "Shares": 0})
            else:
                orders.iloc[j] = pd.Series({"Cash": cash, "Shares": 0})



            print orders
            returnArr.append(cash)
            if i > iterationMin-1:
                converge = 0.2 * returnArr[i - 1]
                if (returnArr[i] - converge) < returnArr[i-1] < (returnArr[i] + converge):
                    break





    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        # Call indicators function to receive indicators with dates before starting range
        normalAverage, bollBand, MACDInd, momentum, prices = plotIndicators(sd - dt.timedelta(days=80), ed, syms)

        # Get indicators for the given dates
        bollBand = bollBand.loc[prices.index >= sd]
        normalAverage = normalAverage.loc[prices.index >= sd]
        MACDInd = MACDInd.loc[prices.index >= sd]
        momentum = momentum.loc[prices.index >= sd]
        prices = prices.loc[prices.index >= sd]

        # Discretize values in the dataframs

        bollBand = self.dfDiscretize(bollBand, steps=self.bins)
        normalAverage = self.dfDiscretize(normalAverage, steps=self.bins)
        MACDInd = self.dfDiscretize(MACDInd, steps=self.bins)
        momentum = self.dfDiscretize(momentum, steps=self.bins)
        firstState = (bollBand.iloc[0][symbol], normalAverage.iloc[0][symbol], MACDInd.iloc[0][symbol], momentum.iloc[0][symbol])
        dims = (self.bins, self.bins, self.bins, self.bins)
        firstState = np.ravel_multi_index(firstState, dims=dims)
        self.learner = ql.QLearner(num_states=self.bins ** 4, num_actions=5,rar=0.98, verbose=False)

        action = self.learner.querysetstate(firstState)
        shares = 0
        orders = pd.DataFrame(columns=["Orders"], index=prices.index)
        for j in range(len(prices.index)):
            # Buy allowed
            if action == 0 and shares in [0, -500]:
                shares += 500
                orders.iloc[j]["Orders"]=500
            elif action == 1 and shares == -500:
                shares += 1000
                orders.iloc[j]["Orders"] = 1000
            # sell allowed
            elif action == 2 and shares in [0, 500]:
                shares -= 500
                orders.iloc[j]["Orders"] = -500
            elif action == 3 and shares == 500:
                shares -= 1000
                orders.iloc[j]["Orders"] = -1000
            else:
                orders.iloc[j]["Orders"] = 0
            nextState = np.ravel_multi_index((bollBand.iloc[j][symbol], normalAverage.iloc[j][symbol],
                                              MACDInd.iloc[j][symbol], momentum.iloc[j][symbol]), dims=dims)

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

    def plotIndicators(self,sd=dt.datetime(2006, 1, 1), ed=dt.datetime(2009, 12, 31), \
                       syms=['IBM'], gen_plot=False):

        # Read in adjusted closing prices for given symbols, date range
        dates = pd.date_range(sd, ed)
        prices_all = get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        # Normalize prices according to first day
        normPrice = prices / prices.iloc[0] - 1

        # Momentum -0.5 - 0.5
        tradingDays = 20
        momentum = (prices / prices.shift(tradingDays)) - 1

        # Simple Moving Average -0.5 - 0.5
        movingAverage = pd.rolling_mean(prices, tradingDays)

        # Prices / SMA
        normalAverage = prices / movingAverage - 1

        # Bollinger Bands  <-1 buy >1 sell
        std = pd.rolling_std(prices, tradingDays)
        lower = movingAverage - 2 * std
        bollBand = (prices - movingAverage) / (2 * std)
        upper = movingAverage + 2 * std
        # print prices.loc['20060706':'20060720']
        # print lower.loc['20060706':'20060720']
        # print bollBand.loc['20060706':'20060720']
        # print bollBand.loc['20080715':'20080727']

        # Exponential Moving Average for 12 days
        ema12 = pd.ewma(prices, com=5.5, min_periods=12)
        # Exponential Moving Average for 26 days
        ema26 = pd.ewma(prices, com=12.5, min_periods=26)
        # MACD
        MACD = ema12 - ema26
        # Signal line to determine trades when MACD crosses it
        signalLine = pd.ewma(MACD, com=4, min_periods=9)
        # Trigger strong momentum and where moving average crosses prices
        MACDInd = MACD - signalLine

        # Compare daily portfolio value with SPY using a normalized plot
        # if gen_plot:
        #     # Generates plots shown in the report
        #     df_MACD = pd.concat([MACD, signalLine], keys=['MACD', 'Signal'], axis=1)
        #     plot_data(df_MACD, title="MACD", ylabel="Normal Price")
        #     df_momentum = pd.concat([momentum, normPrice], keys=['Momentum', 'Normalized Price'], axis=1)
        #     plot_data(df_momentum, title="Momentum", ylabel="Normal Price")
        #     df_SMA = pd.concat([movingAverage / movingAverage.iloc[20], prices / prices.iloc[0], normalAverage],
        #                        keys=['20-day SMA', 'Price', "Price/SMA"], axis=1)
        #     plot_data(df_SMA, title="20-day Simple Moving Average", ylabel="Normal Price")
        #     df_bollBands = pd.concat([prices, upper, lower, movingAverage],
        #                              keys=['Price', 'Upper Band', 'Lower Band', "SMA"], axis=1)
        #     plot_data(df_bollBands, title="Bollinger Bands", ylabel="Normal Price")
        return normalAverage, bollBand, MACDInd, momentum, prices

if __name__=="__main__":
    print "One does not simply think up a strategy"

'''