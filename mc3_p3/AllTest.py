import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import math
from util import get_data, plot_data
import marketsim
import RTLearner as rt
import indicators
import rule_based


def mlTrade(dataX,dataY,testX = None):
    #This MLTrade could be used for both MLTrade and test file
    learner =rt.RTLearner(leaf_size=5)
    learner.addEvidence(dataX,dataY)
    pred = learner.query(dataX)
    if testX==None:
        return pred
    else:
        return learner.query(testX)

def generateFutureOrder(df_ml,index,name="MLBased.csv"):
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
    normalAverage, bollBand, MACDInd, momentum,prices = indicators.plotIndicators(gen_plot=gen_plot)
    normalArray = normalAverage.as_matrix()
    bollBandArray = bollBand.as_matrix()
    MACDIndArray = MACDInd.as_matrix()
    momentumArray = momentum.as_matrix()
    indArrayTrain = np.concatenate((normalArray,bollBandArray,MACDIndArray,momentumArray),axis=1)
    Y = prices.shift(-10)/prices - 1
    YBUY =.03
    YSELL=-.04
    for i in range(Y.shape[0]):
        #print Y.iloc[i]["IBM"]
        if Y.iloc[i]["IBM"]>YBUY:
            Y.iloc[i]["IBM"]=1
        elif Y.iloc[i]["IBM"]<YSELL:
            Y.iloc[i]["IBM"]=-1
        else:
            Y.iloc[i]["IBM"] = 0

    YArray = Y['IBM'].values
    #Build Test Data
    normalAverage, bollBand, MACDInd, momentum,prices = indicators.plotIndicators(sd = dt.datetime(2010,1,4), ed = dt.datetime(2010,12,31),gen_plot=gen_plot)
    normalArray = normalAverage.as_matrix()
    bollBandArray = bollBand.as_matrix()
    MACDIndArray = MACDInd.as_matrix()
    momentumArray = momentum.as_matrix()
    indArrayTest = np.concatenate((normalArray, bollBandArray, MACDIndArray, momentumArray), axis=1)
    orderInfo = mlTrade(indArrayTrain,YArray,indArrayTest)
    orders = generateFutureOrder(orderInfo,prices.index,name="Test.csv")
    sv =100000
    of = "Test.csv"
    portvals = marketsim.compute_portvals(orders_file=of, start_val=sv)
    print portvals.head(1)
    print portvals.tail(1)
    of = "TestOrders.csv"
    rule_based.ordersFile(sd = dt.datetime(2010,1,4), ed = dt.datetime(2010,12,31),normalAverage=normalAverage,\
                          bollBand=bollBand,MACDInd=MACDInd,momentum=momentum,filename=of)
    portvals1 = marketsim.compute_portvals(orders_file=of, start_val=sv)
    print portvals1.head(1)
    print portvals1.tail(1)
    of = "benchmarkTest.csv"
    portvals2 = marketsim.compute_portvals(orders_file=of, start_val=sv)
    print portvals2.head(1)
    print portvals2.tail(1)
    if gen_plot:
        df = pd.concat([portvals2/portvals2.iloc[0], portvals1/portvals1.iloc[0],portvals/portvals.iloc[0]], keys=['Benchmark', 'Rule-Based',"ML-Based"], axis=1)
        ax = df.plot(title="ML Based Trader", fontsize=12,color=["Black","Blue","Green"])
        ax.set_xlabel("Date")
        ax.set_ylabel("Price Normalized")
        plt.show()









if __name__=="__main__":
    main(gen_plot=True)