Created by Dominique Paster
All plots not shown by default must set gen_plot to True.

Part 1: indicators.py
The only function is plotIndicators and it can take the arguments. To show plot make sure to set gen_plot to true. A
sample call is shown below. The function returns return normalAverage,bollBand,MACDInd,momentum,prices

plotIndicators(sd = dt.datetime(2006,1,1), ed = dt.datetime(2009,12,31), \
    syms = ['IBM'],gen_plot=False):
returns normalAverage,bollBand,MACDInd,momentum,prices as dataframes



Part 2: rule_based.py
Uses indicators and marketsim

Two functions main and ordersFile. Main takes in gen_plot to determine whether to make a plot or not. Main function
calls plotIndicators above with all default params to generate indicator values. The calls the ordersFile method to
generate a csv file using the rules. The ordersFile method also returns the DataFrame. Main function then uses marketsim
to get the daily portfolio values.

main(gen_plot=False)

ordersFile(sd = dt.datetime(2006,1,1),ed=dt.datetime(2009, 12, 31), \
                   syms=['IBM'],normalAverage=pd.DataFrame(),bollBand=pd.DataFrame(), \
               MACDInd=pd.DataFrame(),momentum=pd.DataFrame(),filename="orders.csv")
return orders Dataframe and creates csv
Part 3: ML_based.py
Uses RTLearner,marketsim,indicators
There are three functions shown below. The main function calls the other functions to generate data. Main calls indicators
with default params. Then builds matrix and array for mlTrade to use to perform all the machine learning and return an array
The prediction array is fed into generateOrder to create a csv with the default name and then return the create DataFrame.

main(gen_plot=False)

mlTrade(dataX,dataY)
return prediction array
generateOrder(df_ml,index,name="MLBased.csv")
return orders dataframe and creates csv file
Part 4: AllTest.py

This file has same functionality as ML_Based but adds the future dataset ability to the mlTrade function. The functions
are below just in case they are needed.

main(gen_plot=False)
mlTrade(dataX,dataY,testX = None)
generateFutureOrder(df_ml,index,name="MLBased.csv")

