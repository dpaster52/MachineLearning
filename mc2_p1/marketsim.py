"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    # this is the function the autograder will call to test my code
    data = readSortfile(orders_file)
    dates = data.index.get_values()
    symbols = data.Symbol.unique()
    start_date = dates[0]
    end_date = dates[-1]
    #Get values in test range for all symbols
    portvals = get_data(symbols.tolist(), pd.date_range(start_date, end_date))
    portvals= portvals[symbols.tolist()] #remove SPY
    symbols = portvals.columns
    final = pd.DataFrame(index=portvals.index)
    stock = {}
    count = 0
    for symbol in symbols.tolist():
        stock[symbol]=0
    #align columns to match dictionary
    portvals= portvals[stock.keys()]
    final["Value"]=1
    #Loop through days that traded
    for date,symbolDay in portvals.iterrows():
        #Extra credit skip for wife birthday
        if (date == dt.datetime(2011,6,15)):
            #Drop rows in data of trades occur
            if(dt.datetime(2011, 6, 15) in data.index):
                data.drop(dt.datetime(2011, 6, 15),0)

        #check if any trading done this day
        if(date in data.index):
            #action for the given day
            while(date == data.index[count]):
                action = data["Order"].iloc[count]
                #compute daily value for that day
                if (action == "BUY"):
                    #add stock to amount you currently own
                    stock[data["Symbol"].iloc[count]]+= data["Shares"].iloc[count]
                    #calculate money spent on stock
                    symbol= data["Symbol"].iloc[count]
                    start_val-= data["Shares"].iloc[count] * symbolDay.loc[symbol]
                if (action == "SELL"):
                    # add stock to amount you currently own
                    stock[data["Symbol"].iloc[count]] -= data["Shares"].iloc[count]
                    # calculate money spent on stock
                    symbol = data["Symbol"].iloc[count]
                    start_val += data["Shares"].iloc[count] * symbolDay.loc[symbol]
                # leverageTop = (sum((abs(np.array(stock.values())) * symbolDay.tolist())))
                # leverageBottom = (sum((np.array(stock.values()) * symbolDay.tolist())))+start_val
                # leverage = leverageTop/leverageBottom
                # if(leverage>3):
                #     #Undo trade if leverage occurs
                #     if (action == "BUY"):
                #         # add stock to amount you currently own
                #         stock[data["Symbol"].iloc[count]] -= data["Shares"].iloc[count]
                #         # calculate money spent on stock
                #         symbol = data["Symbol"].iloc[count]
                #         start_val += data["Shares"].iloc[count] * symbolDay.loc[symbol]
                #     if (action == "SELL"):
                #         # add stock to amount you currently own
                #         stock[data["Symbol"].iloc[count]] += data["Shares"].iloc[count]
                #         # calculate money spent on stock
                #         symbol = data["Symbol"].iloc[count]
                #         start_val -= data["Shares"].iloc[count] * symbolDay.loc[symbol]
                count +=1
                if (count+1>=data.shape[0]):
                    break
        stock_val =sum((np.array(stock.values()) * symbolDay.tolist()))
        endVal = start_val+stock_val
        final.at[date, 'Value'] = endVal

    final = final.dropna()
    print final
    return final

def compute_portfolio_stats(prices, \
    rfr=0.0, sf=252):
    #Calculate Cumulative return using top and bottom of df and allocation array
    cr =np.sum(((np.array(prices.tail(1))/np.array(prices.head(1))-1)))#*np.array(allocs)))
    #Calculate Average Daily return and Volatility(STD of Daily Return)
    daily= prices.copy()
    daily[1:] = (prices[1:]/prices[:-1].values)-1
    daily = daily[1:]
    adr = np.sum(daily.mean())
    sddr= np.sum(daily.std(axis=0))

    #Calculate the Sharpe ratio
    sr = ((adr-rfr)/sddr)*np.sqrt(sf)

    return cr,adr,sddr,sr

def readSortfile(fileName):
    orders = pd.read_csv(fileName, index_col='Date', parse_dates=True, na_values=['nan'])
    orders = orders.sort_index()
    return orders

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2011,1,14)
    end_date = dt.datetime(2011,8,11)
    portvals1 = get_data(["AAPL"], pd.date_range(start_date, end_date))
    SPY = portvals1['SPY']
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = compute_portfolio_stats(SPY/SPY[1])

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
