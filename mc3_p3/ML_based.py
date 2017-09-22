import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import math
from util import get_data, plot_data
import marketsim
import RTLearner as rt
import BagLearner as bl
import indicators


def mlTrade(dataX,dataY):
    #Create Learner and test on training set
    learner =rt.RTLearner(leaf_size=5)
    #learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=5, boost=False, verbose=False)
    learner.addEvidence(dataX,dataY)
    pred = learner.query(dataX)
    return pred

def generateOrder(df_ml,index,name="MLBased.csv"):
    tenDays = 0
    stock = 0
    ChangePosition = False
    orders = pd.DataFrame(index=index, columns=["Symbol", "Order", "Shares"])
    for i in range(len(df_ml)):
        if (ChangePosition):
            if tenDays>=10:
                ChangePosition=False
                tenDays=0
            else:
                tenDays+=1
                continue
        if(df_ml[i]==1):
            if (stock == 500):
                continue
            stock+=500
            ChangePosition=True
            orders.iloc[i]["Order"]="BUY"
            orders.iloc[i]["Shares"] = "500"
            orders.iloc[i]["Symbol"] = "IBM"
        elif(df_ml[i]==-1):
            if(stock==-500):
                continue
            stock-=500
            ChangePosition=True
            orders.iloc[i]["Order"] = "SELL"
            orders.iloc[i]["Shares"] = "500"
            orders.iloc[i]["Symbol"] = "IBM"
    if stock==500:
        orders.loc[index[-1].date()]["Order"]="SELL"
        orders.loc[index[-1].date()]["Shares"] = "500"
        orders.loc[index[-1].date()]["Symbol"] = "IBM"
    if stock==-500:
        orders.loc[index[-1].date()]["Order"] = "BUY"
        orders.loc[index[-1].date()]["Shares"] = "500"
        orders.loc[index[-1].date()]["Symbol"] = "IBM"
    orders = orders.dropna(how="all")
    orders.to_csv(name, index_label="Date")
    return orders


def main(gen_plot=False):
    #Get values for Indicators
    normalAverage, bollBand, MACDInd, momentum,prices = indicators.plotIndicators(gen_plot=gen_plot)
    normalArray = normalAverage.as_matrix()
    bollBandArray = bollBand.as_matrix()
    MACDIndArray = MACDInd.as_matrix()
    momentumArray = momentum.as_matrix()
    indArray = np.concatenate((normalArray,bollBandArray,MACDIndArray,momentumArray),axis=1)
    #Create future Y predictions
    Y = prices.shift(-10)/prices - 1
    YBUY =.03
    YSELL=-.02
    #Convert predictions to 1, 0, -1 for classifier
    for i in range(Y.shape[0]):
        #print Y.iloc[i]["IBM"]
        if Y.iloc[i]["IBM"]>YBUY:
            Y.iloc[i]["IBM"]=1
        elif Y.iloc[i]["IBM"]<YSELL:
            Y.iloc[i]["IBM"]=-1
        else:
            Y.iloc[i]["IBM"] = 0

    YArray = Y['IBM'].values

    orderInfo = mlTrade(indArray,YArray)
    #create orders file
    orders = generateOrder(orderInfo,prices.index)
    sv =100000
    of = "MLBased.csv"
    portvals = marketsim.compute_portvals(orders_file=of, start_val=sv)
    print portvals.head(1)
    print portvals.tail(1)
    #orders.csv contains data for the rule based trader
    of = "orders.csv"
    portvals1 = marketsim.compute_portvals(orders_file=of, start_val=sv)
    print portvals1.head(1)
    print portvals1.tail(1)
    of = "benchmark.csv"
    portvals2 = marketsim.compute_portvals(orders_file=of, start_val=sv)
    print portvals2.head(1)
    print portvals2.tail(1)
    #generate the plot
    if gen_plot:
        df = pd.concat([portvals2/portvals2.iloc[0], portvals1/portvals1.iloc[0],portvals/portvals.iloc[0]], keys=['Benchmark', 'Rule-Based',"ML-Based"], axis=1)
        ax = df.plot(title="ML Based Trader", fontsize=12,color=["Black","Blue","Green"])
        ax.set_xlabel("Date")
        ax.set_ylabel("Price Normalized")
        stock = 0
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









if __name__=="__main__":
    main(gen_plot=False)