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
    return Dataframe_to_return,sheet_name


rp = rz.Plotting(True)
Frequency = ("d","m")
#Take static data
Economic_Data,list_name = union_sheets_value(pd . read_excel('Economy_Data.xlsx',sheet_name=None),'Economy_Data.xlsx')
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
    stock_names.append('GDAXI_'+f)
        
    """
    Calculating LogLevel and LogPrice for market and stocks
    """
    df_eco_ret = pd.DataFrame(data=(100 *( np . log ( Economic_Data ) -np . log ( Economic_Data . shift (1) ))), columns = list_name)
    df_equity_ret = pd.DataFrame(data=(100 *( np . log ( temp_stock ) -np . log ( temp_stock . shift (1) ))),columns = stock_names)
    df_equity_L =pd.DataFrame(data=(np.log(temp_stock)),columns=stock_names)

    """
    End
    """

    #when the plot beeing print, the file  is called LogLevel_Correlation...
    rp.plot_line(df_equity_ret,time_series,f) 
    rp.plot_line2(df_equity_L,time_series,f)
    rp.plot_simple(Economic_Data,"Monthly")

    """
    FATTO 2
    """
    """
    Correlation plots
    """
    rp.plot_correlation(df_equity_ret.dropna(), 20, f,"ret")
    rp.plot_correlation(df_eco_ret.dropna(), 20, f,"Eco_Ret") 
    rp.plot_correlation(df_equity_L.dropna(), 20, f,"log")     
    rp.plot_correlation(Economic_Data.dropna(), 20,"Monthly","Eco")  
    
    """
    Table
    """
    
    table_log = rf.table(df_equity_L)

    table_eco = rf.table(Economic_Data)
    table_log.to_excel("table_log"+f+".xlsx")
    table_eco.to_excel("table_eco.xlsx")

    """
    ADF TEST 
    """

    #delete last 24 month and delete null value or delete 5% of daily
    n = int(np.round(df_equity_L.dropna().shape[0] * 0.05) ) if f == "d" else 24
    nlag = 22 if f == "d" else 12
    #Cuts
    df_eco_cut = Economic_Data.iloc[:-24,:]
    df_eco_ret_cut = df_eco_ret.dropna().iloc[:-24,:]
    df_log_cut = df_equity_L.dropna().iloc[:-n,:]
    df_ret_cut = df_equity_ret.dropna().iloc[:-n,:]
    
    adf_log = rf.adf_test(df_log_cut,nlag)
    adf_ret=rf.adf_test(df_ret_cut,nlag)
    adf_eco=rf.adf_test(df_eco_cut,nlag)
    adf_eco_ret_cut=rf.adf_test(df_eco_ret_cut,nlag)

    adf_log.to_excel("pippo.xlsx")

    #to excel, eco and log
    #SE sono stazionari fare arma altrimenti renderli stazionari
    ret_stat,ret_non_stat=rf.stationarity_and_not_stationary(adf_log)
    eco_stat,eco_non_stat=rf.stationarity_and_not_stationary(adf_eco)
    eco_ret_stat,eco_ret_non_stat=rf.stationarity_and_not_stationary(adf_eco_ret_cut)



    for i in adf_log.columns:
        rp.plotbar(adf_log[i],f,"Log") 
    for i in adf_eco.columns:
        rp.plotbar(adf_eco[i],f,"Log")
    

    #First difference
    df_ret_cut = df_ret_cut.loc[:,ret_non_stat]

    for i in ret_stat:
        df_ret_cut[i] = df_log_cut.iloc[:-n,:][i]

    df_eco_ret_cut = df_eco_ret_cut.loc[:,eco_ret_non_stat]

    for i in eco_ret_stat:
         df_eco_ret_cut[i] = df_eco_cut[i]
    

    adf_ret=rf.adf_test(df_ret_cut.dropna(),nlag)

    adf_eco_ret_cut=rf.adf_test(df_eco_ret_cut.dropna(),nlag)
    """
    rp.plotbar(df_ret_cut["pvalue"],f,"ret")
    rp.plotbar(adf_eco_ret_cut["pvalue"],f,"ret")
    """
    rp.histo_plot2(df_log_cut,df_ret_cut,f)
    rp.histo_plot2(df_eco_cut,df_eco_ret_cut,f,90)
    
    """
    rf.jungbox_test(rf.test(df_log_cut),10)
    """

    """
    ACF E PACF RESIDUI
    """

    
