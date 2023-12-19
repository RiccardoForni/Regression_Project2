import numpy as np
import pandas as pd
import Regression_function as rf
import Regression_Plotting as rz

rp = rz.Plotting(False)
sheets={"EURIBOR_3_M":"BD INTEREST RATES - EURIBOR RATE - 3 MONTH NADJ"}
Frequency = ("m","d")
for f in Frequency:
    
    Market= pd . read_excel ('Gmarket.xlsx', 'GDAXI_'+f)
    Interest_Rate = pd . read_excel ('Gmarket.xlsx', 'Bund')
    Stocks = pd . read_excel ('Stocks.xlsx', sheet_name=None)

    time_series=Market.iloc[:,0] #Date series

    j = lambda d : [s for s in d.keys() if '_'+f in s]
    
    stock_names = list(j(Stocks))
    Market = Market.iloc[:,1]

    temp_stock = pd.DataFrame()
    i = 0
    for s in stock_names:
        Stock = pd.read_excel('Stocks.xlsx',s)
        temp_stock.insert(i,s,Stock.iloc[:,1])
        
        i = i+1
        

    df_factors = pd.DataFrame(data=(100 *np . log ( Market ) -np . log ( Market . shift (1) )))
    df_factors.insert(0,"LogLevel",np . log ( Market ))
    df_factors.columns = df_factors.columns.str.replace("Close", "LogPrice")



    df_stocks = pd.DataFrame(data=(100 *( np . log ( temp_stock ) -np . log ( temp_stock . shift (1) ))),columns = stock_names)
    df_stocks_L =pd.DataFrame(data=(np.log(temp_stock)),columns=stock_names)

    

    """
    df_stocks = df_stocks.iloc[1:,]
    df_stocks.index = df_factors.index = time_series
    """

    rp.plot_line(df_stocks_L,time_series,f)
    rp.plot_line(df_factors["LogLevel"],time_series,f)
    adfullerstocks=rf.adf_test(df_stocks_L.dropna(),21)
    adfullermarket=rf.adf_test(df_factors["LogLevel"].dropna(),21)
    """
    LogPrice
    """
    rp.plot_line2(df_stocks,time_series,f)
    rp.plot_line2(df_factors["LogPrice"],time_series,f)
    adfullerstocks=rf.adf_test(df_stocks.dropna(),21)
    adfullermarket=rf.adf_test(df_factors["LogPrice"].dropna(),21)
    
    """ TESTING IGNORE IT
    rp.test(rf.jacque(df_stocks_L),f)
    """ 
