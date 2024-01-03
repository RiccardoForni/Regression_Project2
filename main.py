import numpy as np
import pandas as pd
import Regression_function as rf
import Regression_Plotting as rz

def union_sheets_value(Dataframe,name_xlsx,frequency = ""):
     
    j = lambda d,f : [s for s in d.keys() if '_'+f in s or s in s]
    sheet_name = list(j(Dataframe,frequency))
    Dataframe_to_return = pd.DataFrame()
    i = 0
    for s in sheet_name:
            Dataframe = pd.read_excel(name_xlsx,s)
            Dataframe_to_return.insert(i,s,Dataframe.iloc[:,1])
            i = i+1
    return Dataframe_to_return


rp = rz.Plotting(False)
Frequency = ("m","d")
#Take static data
Economic_Data = union_sheets_value(pd . read_excel('Economy_Data.xlsx',sheet_name=None),'Economy_Data.xlsx')
Interest_Rate = pd . read_excel ('Gmarket.xlsx', 'Bund')
Stocks = pd . read_excel ('Stocks.xlsx', sheet_name=None)


for f in Frequency:
    
    #Take market index depends on frequency
    Market= pd . read_excel ('Gmarket.xlsx', 'GDAXI_'+f)
    time_series=Market.iloc[:,0] #Date series depends on frequency

    #Create list with stocks name from sheet list(d for daily series and m for monthly series)
    j = lambda d,f : [s for s in d.keys() if '_'+f in s]
    stock_names = list(j(Stocks,f))
    Market = Market.iloc[:,1]

    temp_stock = pd.DataFrame()
    i = 0
    for s in stock_names:
        Stock = pd.read_excel('Stocks.xlsx',s)
        temp_stock.insert(i,s,Stock.iloc[:,1])
        
        i = i+1
    temp_stock.insert(i,'GDAXI_'+f,Market)
        
    """
    Calculating LogLevel and LogPrice for market and stocks
    """

    df_equity_ret = pd.DataFrame(data=(100 *( np . log ( temp_stock ) -np . log ( temp_stock . shift (1) ))),columns = stock_names)
    df_equity_L =pd.DataFrame(data=(np.log(temp_stock)),columns=stock_names)

    """
    End
    """

    #when the plot beeing print, the file is called LogLevel_Correlation...
    rp.plot_line(df_equity_L,time_series,f) 
    rp.plot_simple(Economic_Data,"Monthly")
    """
    Correlation plots
    """
    rp.plot_correlation(df_equity_L, 20, f)     
    rp.plot_correlation(Economic_Data, 20,"Monthly")  

    df_m_eco_cut = Economic_Data.iloc[:-24,:]
    """
    if f == "m":
         df_m_cut = df_equity_L.iloc[:-24,:]
         df_ret_m_cut = df_equity_ret.iloc[:-24,:]
         adfullerstocks=rf.adf_test(df_m_cut,21)
         adf_ret_monthly_cut = rf.adf_test(df_ret_m_cut,21)
    else:
         n = int(np.round(df_equity_L.dropna().shape[0] * 0.4) )
         df_ret_d_cut = df_equity_ret.iloc[:-24,:]
         df_equity_L_d_cut = df_equity_L.iloc[:-n,:] 
         adfullerstocks=rf.adf_test(df_equity_L_d_cut,21)
         adf_ret_daily_cut = rf.adf_test(df_equity_L_d_cut,21)

    adfullereco=rf.adf_test(df_m_eco_cut,21)

    """

    

    """
    adfullerstocks=rf.adf_test(df_stocks_L.dropna(),21)
    adfullermarket=rf.adf_test(df_factors["LogLevel"].to_frame().dropna(),21)
    """
    
    """
    rp.plot_line2(df_stocks,time_series,f)
    rp.plot_line2(df_factors["LogPrice"].to_frame(),time_series,f)
    adfullerstocks=rf.adf_test(df_stocks.dropna(),21)
    adfullermarket=rf.adf_test(df_factors["LogPrice"].to_frame().dropna(),21)
    
    df_stock_L_jacque = rf.jacque(df_stocks_L.dropna())
    df_stock_jacque = rf.jacque(df_stocks.dropna())

    df_Market_jacque = rf.jacque(df_factors.dropna())

    rp.plot_line3(df_stocks,f)

    rp.prob_plot(df_stocks,f)
    

    rp.histo_plot(df_stocks,f)
    """
