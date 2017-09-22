"""MC1-P2: Optimize a portfolio."""
"""Dominique Paster 2:31PM"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import scipy.optimize as spo
from util import get_data, plot_data

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=True):
    #Constraints to make sure allocations are not too large
    cons = ({ 'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs) })
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    allocs = np.asarray([0.1,0.4,0.2,0.2,0.1])
    bound=((0,1),(0,1),(0,1),(0,1),(0,1))
    sr = spo.minimize(sr_val,allocs,args=(prices,),method='SLSQP',\
            bounds=bound,constraints=cons,options={'disp':True})
    allocs = sr.x #get the optimized allocations from funtion call above
    sr = sr.fun *-1 #fix value to be the max
    port_val = np.divide(prices,prices.head(1)) #normalize data
    port_val= port_val *np.array(allocs) #Apply the allocations
    port_val = np.sum(port_val,axis=1) #sum values to get total portfolio
    cr, adr, sddr = compute_portfolio_stats(port_val) #compute  statistics
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, np.divide(prices_SPY,prices_SPY.head(1))], keys=['Portfolio', 'SPY'], axis=1)
        graph = df_temp.plot(title= "Daily Portfolio value and SPY")
        graph.set_xlabel("Date")
        graph.set_ylabel("Normalized price")
        plt.grid()
        plt.show()
        pass

    return allocs, cr, adr, sddr, sr
def compute_portfolio_stats(prices):
        #Calculate Cumulative return using top and bottom of df and allocation array
        cr = np.sum(((np.array(prices.tail(1)) / np.array(prices.head(1)) - 1)))
        # Calculate Average Daily return and Volatility(STD of Daily Return)
        daily = prices.copy()
        daily[1:] = (prices[1:] / prices[:-1].values) - 1
        daily = daily[1:]
        # print(daily)
        adr = np.sum(daily.mean())
        sddr = np.sum(daily.std(axis=0))
        return cr, adr, sddr


def sr_val(allocs, data):
    # allocs is a array with 4 values
    # data where each row is the adjusted close for that time period
    rfr = 0
    sf = 252.0
    prices = np.sum(np.multiply(np.divide(data, data.head(1)), np.array(allocs)), axis=1)
    daily = prices.copy()
    daily[1:] = (prices[1:] / prices[:-1].values) - 1
    daily = daily[1:]
    # print(daily)
    adr = np.sum(daily.mean())
    sddr = np.sum(daily.std(axis=0))
    # Calculate the Sharpe ratio
    sr = ((adr - rfr) / sddr) * np.sqrt(sf) * -1
    return sr


def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code
    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    symbols = ['IBM', 'X', 'HNZ', 'XOM', 'GLD']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date, \
                                                        syms=symbols, \
                                                        gen_plot=True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
