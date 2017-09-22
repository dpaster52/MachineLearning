# simple moving average price
# bolinger bands
# MACD
# RSI


import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import indicators
import math
from util import get_data
import marketsim


def ordersFile(sd = dt.datetime(2006,1,1),ed=dt.datetime(2009, 12, 31), \
                   syms=['IBM'],normalAverage=pd.DataFrame(),bollBand=pd.DataFrame(), \
               MACDInd=pd.DataFrame(),momentum=pd.DataFrame(),filename="orders.csv"):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    orders = pd.DataFrame(index=prices.index, columns=["Symbol","Order","Shares"])

    #Create boolean values to make decisions on
    #MACD
    alertMACDBuy =MACDInd<0
    alertMACDSell=MACDInd>0
    #Bollinger Band Values
    bollBandBuy = bollBand < -1
    bollBandSell = bollBand > 1
    #Momentum Values
    momentumBuy = momentum > 0
    momentumSell = momentum < 0
    #price/SMA values
    normalAverageSell = normalAverage > 0
    normalAverageBuy = normalAverage < 0
    #Loop variable to determine when you can trade
    tenDays =0
    ChangePosition = False
    count = -1
    stock=0
    #Loop through trading days
    for date,symbol in prices.iterrows():
        t = date.date()
        count+=1
        if (ChangePosition):
            if tenDays>=10:
                ChangePosition=False
                tenDays=0
            else:
                tenDays+=1
                continue
        if (bollBandBuy.loc[t]['IBM'] or (momentumBuy.loc[t]['IBM'] and normalAverageBuy.loc[t]['IBM'])):
            if not alertMACDBuy.loc[t]["IBM"]:
                if(count>(26+9)):
                    if alertMACDBuy.iloc[count+1]["IBM"] or alertMACDBuy.iloc[count+2]["IBM"]:
                        #print str(t)
                        continue
            if (stock == 500):
                continue
            stock+=500
            ChangePosition=True
            orders.loc[date.date()]["Order"]="BUY"
            orders.loc[date.date()]["Shares"] = "500"
            orders.loc[date.date()]["Symbol"] = syms[0]
        elif(bollBandSell.loc[t]['IBM'] or (momentumSell.loc[t]['IBM'] and normalAverageSell.loc[t]['IBM'])):
            if not alertMACDSell.loc[t]["IBM"]:
                if (count > (26 + 9)):
                    if alertMACDSell.iloc[count + 1]["IBM"] or alertMACDSell.iloc[count + 2]["IBM"]:
                        #print str(t)
                        continue
            if(stock==-500):
                continue
            stock-=500
            ChangePosition=True
            orders.loc[date.date()]["Order"] = "SELL"
            orders.loc[date.date()]["Shares"] = "500"
            orders.loc[date.date()]["Symbol"] = syms[0]



    # Add value to end stock position on end date
    if stock==500:
        orders.loc[ed]["Order"]="SELL"
        orders.loc[ed]["Shares"] = "500"
        orders.loc[ed]["Symbol"] = syms[0]
    if stock==-500:
        orders.loc[ed]["Order"] = "BUY"
        orders.loc[ed]["Shares"] = "500"
        orders.loc[ed]["Symbol"] = syms[0]

    #Create orders file and return DataFrame
    orders =orders.dropna(how="all")
    orders.to_csv(filename,index_label="Date")
    return orders

def main(gen_plot=False):

    of = "benchmark.csv"
    sv = 100000
    #Get Technical Values from indicators file
    normalAverage, bollBand, MACDInd,momentum,prices = indicators.plotIndicators(gen_plot=gen_plot)
    orders = ordersFile(normalAverage=normalAverage, bollBand=bollBand, MACDInd=MACDInd, momentum=momentum)
    # Process orders
    portvals = marketsim.compute_portvals(orders_file=of, start_val=sv)
    #Print statements to determine start and end value
    print portvals.head(1)
    print portvals.tail(1)
    of = "orders.csv"
    #default name of file is orders.csv
    portvals1 = marketsim.compute_portvals(orders_file=of, start_val=sv)
    print portvals1.head(1)
    print portvals1.tail(1)
    if gen_plot:
        df = pd.concat([portvals/portvals.iloc[0], portvals1/portvals1.iloc[0]], keys=['Benchmark', 'Rule-Based'], axis=1)
        ax = df.plot(title="Rule Based Trader", fontsize=12,color=["Black","Blue"])
        ax.set_xlabel("Date")
        ax.set_ylabel("Price Normalized")
        stock = 0
        #Loop through orders to add vertical lines
        for date,order in orders.iterrows():
            if order.loc["Order"]=="SELL":
                stock-=500
                if stock==0:
                    ax.axvline(date.date(),color="Black")
                else:
                    ax.axvline(date.date(), color="Red")
            else:
                stock+=500
                if stock==0:
                    ax.axvline(date.date(),color="Black")
                else:
                    ax.axvline(date.date(), color="Green")
        plt.show()

if __name__ == "__main__":
    main(gen_plot=False)
