#simple moving average price
#bolinger bands
#MACD
#RSI


import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import math
from util import get_data, plot_data
import time

def plotIndicators(sd = dt.datetime(2006,1,1), ed = dt.datetime(2009,12,31), \
    syms = ['IBM'],gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    # Normalize prices according to first day
    normPrice = prices / prices.iloc[0]-1

    
    # Momentum -0.5 - 0.5
    tradingDays = 20
    momentum = (prices / prices.shift(tradingDays)) - 1

    # Simple Moving Average -0.5 - 0.5
    movingAverage = pd.rolling_mean(prices,tradingDays)

    #Prices / SMA
    normalAverage = prices/movingAverage-1

    # Bollinger Bands  <-1 buy >1 sell
    std = pd.rolling_std(prices,tradingDays)
    lower = movingAverage - 2 *std
    bollBand = (prices-movingAverage)/(2*std)
    upper = movingAverage + 2 *std
    #print prices.loc['20060706':'20060720']
    #print lower.loc['20060706':'20060720']
    #print bollBand.loc['20060706':'20060720']
    #print bollBand.loc['20080715':'20080727']

    #Exponential Moving Average for 12 days
    ema12 = pd.ewma(prices,com=5.5,min_periods=12)
    # Exponential Moving Average for 26 days
    ema26 = pd.ewma(prices, com=12.5, min_periods=26)
    #MACD
    MACD = ema12-ema26
    #Signal line to determine trades when MACD crosses it
    signalLine = pd.ewma(MACD,com=4,min_periods=9)
    #Trigger strong momentum and where moving average crosses prices
    MACDInd = MACD - signalLine
    
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # Generates plots shown in the report
        df_MACD = pd.concat([MACD,signalLine], keys=['MACD','Signal'], axis=1)
        plot_data(df_MACD , title="MACD", ylabel="Normal Price")
        df_momentum = pd.concat([momentum, normPrice], keys=['Momentum', 'Normalized Price'], axis=1)
        plot_data(df_momentum, title="Momentum", ylabel="Normal Price")
        df_SMA = pd.concat([movingAverage/movingAverage.iloc[20], prices/prices.iloc[0],normalAverage], keys=['20-day SMA', 'Price',"Price/SMA"], axis=1)
        plot_data(df_SMA, title="20-day Simple Moving Average", ylabel="Normal Price")
        df_bollBands = pd.concat([prices,upper,lower,movingAverage], keys=['Price','Upper Band', 'Lower Band',"SMA"], axis=1)
        plot_data(df_bollBands, title="Bollinger Bands", ylabel="Normal Price")
    return normalAverage,bollBand,MACDInd,momentum,prices



if __name__ == "__main__":
    plotIndicators()
