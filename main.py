import numpy as np
import pandas as pd

import Regression_function as rf
import Regression_Plotting as rz

def union_sheets_value(Dataframe,name_xlsx,frequency = ""):
     
    j = lambda d,f : [s for s in d.keys() if '_'+f in s or s in s]
    sheet_name = list(j(Dataframe,frequency))
    Dataframe_to_return = pd.DataFrame()
    Dataframe = pd.read_excel(name_xlsx,"cpi")
    dates = Dataframe.iloc[:,0]
    i = 0
    for s in sheet_name:
            Dataframe = pd.read_excel(name_xlsx,s)
            Dataframe_to_return.insert(i,s,Dataframe.iloc[:,1])
            i = i+1
    return Dataframe_to_return,sheet_name,dates


rp = rz.Plotting(True)
Frequency = ("d","m")
#Take static data
Economic_Data,list_name,datess = union_sheets_value(pd . read_excel('Economy_Data.xlsx',sheet_name=None),'Economy_Data.xlsx')
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

    if f == "d":
        df_equity_ret_squared = df_equity_ret ** 2

    """
    End
    """

    #when the plot beeing print, the file  is called LogLevel_Correlation...
    rp.plot_line(df_equity_ret,time_series,f) 
    rp.plot_line2(df_equity_L,time_series,f)
    

    """
    FATTO 2
    """
    """
    Correlation plots
    """
    rp.plot_correlation(df_equity_ret.dropna(), 20, f,"ret")
    rp.plot_correlation(df_eco_ret.dropna(), 20,"m","Eco_Ret") 
    rp.plot_correlation(df_equity_L.dropna(), 20, f,"log")     

    
    """
    Table
    """
    
    table_log = rf.table(df_equity_L)

    
    table_log.to_excel("table_log"+f+".xlsx")


    """
    ADF TEST 
    """
            
    #delete last 24 month and delete null value or delete last 6 month of daily
    n = int(np.round(df_equity_L.dropna().shape[0] * 0.05) ) if f == "d" else 24
    nlag = 22 if f == "d" else 12
    #Cuts

    df_log_cut = df_equity_L.dropna().iloc[:-n,:]
    df_ret_cut = df_equity_ret.dropna().iloc[:-n,:]
    time_stock = time_series.values[0]


    nnlg = 20


    adf_log = rf.adf_test(df_log_cut,nlag)
    log_stat,log_non_stat=rf.stationarity_and_not_stationary(adf_log)
    print(log_non_stat)
    old_non_stat = log_non_stat 
    adf_log.to_excel(rz.folder_definer(str(4)+"excel")+"adf_log_"+f+".xlsx")
    for i in adf_log.columns:
        rp.plotbar(adf_log[i],f,str(4)+"Log") 

    df_ret_cut = df_ret_cut.loc[:,log_non_stat]
    #adf first_diff
    adf_ret=rf.adf_test(df_ret_cut,nlag)

    adf_ret.to_excel(rz.folder_definer(str(4)+"excel_ret")+"adf_ret_"+f+".xlsx")

    #to excel, eco and log
    #SE sono stazionari fare arma altrimenti renderli stazionari
    first_ret_stat,first_ret_non_stat=rf.stationarity_and_not_stationary(adf_ret)

    for i in adf_ret.columns:
        rp.plotbar(adf_ret[i],f,str(4)+"ret")


    """
    Arma
    """
    df_ret_cut = df_ret_cut.loc[:,first_ret_stat]
    for i in log_stat:
        df_ret_cut[i] = df_equity_L.dropna().iloc[:-n,:][i]

    arma_ret = rf.arma(df_ret_cut.dropna(how="all"), str.upper(f), time_stock,
                            maxlag = 3,
                            criterion = "AIC")
    arma_ret.to_excel(rz.folder_definer(str(4)+"excel_arma")+"arma"+f+".xlsx")
    
    for i in arma_ret.index:
        print(i)
        print(arma_ret.loc[i,"table"])
    rp.resid_graph(arma_ret,f+"_"+str(4), nlg = nnlg)

    j_ret=rf.jungbox_test(arma_ret["resid"],10)
    j_ret.to_excel(rz.folder_definer(str(4)+"excel_jbox")+"ret_jbox"+f+".xlsx")

    for i,y in zip(j_ret.index,j_ret.columns):
        rp.plotbar(pd.Series(j_ret.loc[i,y],name = y),f,str(4)+"j_ret") 
        


    #POINT 5
    #Taking the last two years and last six month
    shape=df_equity_L.dropna().shape[0]
    
    n = int(np.round(shape * 0.05) ) if f == "d" else 24
    nlag = 8 if f == "d" else 4
    
    #Cuts
    
    df_log_cut = df_equity_L.dropna().tail(n)
    df_ret_cut = df_equity_ret.dropna().tail(n)


    time_stock = time_series.tail(n).values[0]
    nnlg = 4
    print(f)
    

    adf_log = rf.adf_test(df_log_cut,nlag)
    log_stat,log_non_stat=rf.stationarity_and_not_stationary(adf_log)

    df_log_cut = df_log_cut.loc[:,log_stat]


    arma_log = rf.arma(df_log_cut.dropna(how="all"), str.upper(f), time_stock,
                            maxlag = 3,
                            criterion = "AIC")
    time_stock=time_series.tail(24).values[0]
    df_forecast = {}
    
    
    for i in arma_log.index:
        forecast,index = rf.forecast(df_equity_L.dropna(how="all")[i], arma_log.loc[i], df_log_cut[i], f,time_stock,n_f = n)
        df_forecast[i]=rp.plot_forecast(forecast.dropna(how="all"),df_log_cut.dropna(how="all")[i],time_stock,f,i)
    
    df_log_non_stat = df_equity_L.dropna().loc[:,old_non_stat]
    df_level_non_stat = temp_stock.dropna().loc[:,old_non_stat]

    df_log_non_stat_cut = df_equity_L.dropna().tail(n).loc[:,old_non_stat]
    df_level_non_stat_cut = temp_stock.dropna().tail(n).loc[:,old_non_stat]
    
    arma_log = rf.arma(df_log_non_stat_cut.dropna(how="all"), str.upper(f), time_stock,
                            maxlag = 3,
                            criterion = "AIC")
    
    arma_level = rf.arma(df_level_non_stat_cut.dropna(how="all"), str.upper(f), time_stock,
                            maxlag = 3,
                            criterion = "AIC")

    print(old_non_stat)
    print(df_log_non_stat)
    lo=[]
    ll=[]
    df_next = []
    
    for i in df_log_non_stat.columns:
        forecast,index = rf.forecast(df_log_non_stat[i], arma_log.loc[i], df_log_non_stat_cut[i], f,time_stock,n_f = n)
        forecast["Prediction"].to_excel("passing.xlsx")
        Dataframe = pd.read_excel("passing.xlsx")
        Dataframe = Dataframe.iloc[:,1].astype(np.float64)
        
        forecast,index = rf.forecast(Dataframe, arma_log.loc[i], df_log_non_stat_cut[i], f,time_stock,n_f = n,select=True)

        df_next.append(rp.plot_forecast(forecast.dropna(how="all"),df_log_non_stat_cut.dropna(how="all")[i],time_stock,f,i+"forecast_of_forecast_log"))
        print("ok")

    for i in df_level_non_stat.columns:
        forecast,index = rf.forecast(df_level_non_stat[i], arma_level.loc[i], df_level_non_stat_cut[i], f,time_stock,n_f = n)
        forecast["Prediction"].to_excel("passing.xlsx")
        Dataframe = pd.read_excel("passing.xlsx")
        Dataframe = Dataframe.iloc[:,1].astype(np.float64)
        forecast,index = rf.forecast(Dataframe, arma_level.loc[i], df_level_non_stat_cut[i], f,time_stock,n_f = n,select=True)
        df_next.append(rp.plot_forecast(forecast.dropna(how="all"),df_level_non_stat.dropna(how="all")[i],time_stock,f,i+"forecast_of_forecast_level"))
    

    print("stop")

    
    if f=="d":
        
        n = int(np.round(df_equity_ret_squared.dropna().shape[0] * 0.05) ) if f == "d" else 24
        nlag = 22 if f == "d" else 12
        #Cuts
        nnlg = 20

        df_equity_ret_squared_cut = df_equity_ret_squared.dropna().iloc[:-n,:]
        time_stock = time_series.values[0]

        adf_equity_ret_squared_cut = rf.adf_test(df_equity_ret_squared_cut,nlag)
        squared_stat,squared_non_stat=rf.stationarity_and_not_stationary(adf_equity_ret_squared_cut)

        adf_equity_ret_squared_cut.to_excel(rz.folder_definer(str(4)+"excel")+"adf_equity_ret_squared_cut.xlsx")

        for i in adf_equity_ret_squared_cut.columns:
            rp.plotbar(adf_equity_ret_squared_cut[i],f,str(4)+"squared_ret") 


        df_equity_ret_squared_cut = df_equity_ret_squared_cut.loc[:,squared_stat]

  
        arma_ret_squared = rf.arma(df_equity_ret_squared_cut.dropna(how="all"), str.upper("d"), time_stock,
                            maxlag = 3,
                            criterion = "AIC")

        arma_ret_squared.to_excel(rz.folder_definer(str(4)+"excel_arma")+"arma_ret_squared"+f+".xlsx")
        rp.resid_graph(arma_ret_squared,f+"_squared_"+str(4), nlg = nnlg)

        j_ret_squared=rf.jungbox_test(arma_ret_squared["resid"],10)
        j_ret_squared.to_excel(rz.folder_definer(str(4)+"excel_jbox")+"j_ret_squared"+f+".xlsx")

        for i,y in zip(j_ret_squared.index,j_ret_squared.columns):
            rp.plotbar(pd.Series(j_ret.loc[i,y],name = y),f,str(4)+"j_ret_squared") 


        
        df_equity_ret_squared_cut = df_equity_ret_squared.dropna().tail(n)
        time_stock=time_series.values[0]
        for i in arma_ret_squared.index:

                forecast,index = rf.forecast(df_equity_ret_squared.dropna(how="all")[i], arma_ret_squared.loc[i], df_equity_ret_squared_cut[i], f,time_stock,n_f = n)
                    
                rp.plot_forecast(forecast.dropna(how="all"),df_equity_ret_squared_cut[i],time_stock,f,i+"squared")                   
    


"""
Point 4
"""

rp.plot_simple(Economic_Data,"Monthly")
rp.plot_correlation(Economic_Data.dropna(), 20,"Monthly","Eco")  


table_eco = rf.table(Economic_Data)
table_eco.to_excel("table_eco.xlsx")

df_eco_cut = Economic_Data.iloc[:-24,:]
df_eco_ret_cut = df_eco_ret.dropna().iloc[:-24,:]
time_eco = time_series.values[0]
nlag= 12
nnlg = 20

adf_eco=rf.adf_test(df_eco_cut,nlag)
adf_eco.to_excel(rz.folder_definer(str(4)+"excel")+"adf_eco.xlsx")

for i in adf_eco.columns:
    rp.plotbar(adf_eco[i],f,str(4)+"eco")

eco_stat,eco_non_stat=rf.stationarity_and_not_stationary(adf_eco)
old_non_stat = eco_non_stat
df_eco_ret_cut = df_ret_cut.loc[:,eco_non_stat]
adf_eco_ret_cut=rf.adf_test(df_eco_ret_cut,nlag)

adf_eco_ret_cut.to_excel(rz.folder_definer(str(4)+"excel")+"adf_eco_ret_cut.xlsx")
for i in adf_eco_ret_cut.columns:
    rp.plotbar(adf_eco_ret_cut[i],f,str(4)+"eco_ret")

eco_ret_stat,eco_ret_non_stat=rf.stationarity_and_not_stationary(adf_eco_ret_cut)
df_eco_ret_cut = df_eco_ret_cut.loc[:,eco_ret_non_stat]

df_eco_ret_cut =  df_eco_ret_cut.loc[:,eco_ret_stat]
for i in eco_ret_stat:
        df_ret_cut[i] = df_eco_ret.dropna().iloc[:-24,:]

arma_eco = rf.arma(df_eco_ret_cut.dropna(how="all"),"M",time_eco,
                        maxlag = 3,
                        criterion = "AIC")
arma_eco.to_excel(rz.folder_definer(str(4)+"excel_arma")+"arma_eco.xlsx")


rp.resid_graph(arma_eco, "Monthly"+"_"+str(4), nlg = nnlg)

j_eco=rf.jungbox_test(arma_eco["resid"],10)

j_eco.to_excel(rz.folder_definer(str(4)+"excel_jbox")+"eco_jbox"+f+".xlsx")

for i,y in zip(j_eco.index,j_eco.columns):
        rp.plotbar(pd.Series(j_eco.loc[i,y],name =  y),f,str(4)+"j_eco") 
        



df_eco_cut = Economic_Data.tail(24)
df_eco_ret_cut = df_eco_ret.dropna().tail(24)
time_eco = time_series.tail(24).values[0]



df_forecast = []
for i in arma_eco.index:

    forecast,index = rf.forecast(df_eco_ret.dropna()[i], arma_eco.loc[i], df_eco_ret_cut[i], f,time_eco,
                            n_f = 24)
    df_forecast.append(rp.plot_forecast(forecast.dropna(),df_eco_ret_cut[i],time_eco,f,i+"eco"))


   
df_log_non_stat = df_eco_ret.dropna().loc[:,old_non_stat]
df_level_non_stat = Economic_Data.dropna().loc[:,old_non_stat]

df_log_non_stat_cut = df_eco_ret.dropna().tail(n).loc[:,old_non_stat]
df_level_non_stat_cut = Economic_Data.dropna().tail(n).loc[:,old_non_stat]
arma_log = rf.arma(df_log_non_stat_cut.dropna(how="all"), str.upper(f), time_stock,
                            maxlag = 3,
                            criterion = "AIC")
    
arma_level = rf.arma(df_level_non_stat_cut.dropna(how="all"), str.upper(f), time_stock,
                            maxlag = 3,
                            criterion = "AIC")

lo=[]
ll=[]

   
for i in df_log_non_stat.columns:
    forecast,index = rf.forecast(df_log_non_stat[i], arma_log.loc[i], df_log_non_stat_cut[i], f,time_eco,n_f = n)
    forecast["Prediction"].to_excel("passing.xlsx")
    Dataframe = pd.read_excel("passing.xlsx")
    Dataframe = Dataframe.iloc[:,1].astype(np.float64)
    
    forecast,index = rf.forecast(Dataframe, arma_log.loc[i], df_log_non_stat_cut[i], f,time_eco,n_f = n,select=True)

    df_next.append(rp.plot_forecast(forecast.dropna(how="all"),df_log_non_stat_cut.dropna(how="all")[i],time_eco,f,i+"eco_forecast_of_forecast_log"))
    print("ok")

for i in df_level_non_stat.columns:
    forecast,index = rf.forecast(df_level_non_stat[i], arma_level.loc[i], df_level_non_stat_cut[i], f,time_eco,n_f = n)
    forecast["Prediction"].to_excel("passing.xlsx")
    Dataframe = pd.read_excel("passing.xlsx")
    Dataframe = Dataframe.iloc[:,1].astype(np.float64)
    forecast,index = rf.forecast(Dataframe, arma_level.loc[i], df_level_non_stat_cut[i], f,time_eco,n_f = n,select=True)
    df_next.append(rp.plot_forecast(forecast.dropna(how="all"),df_level_non_stat.dropna(how="all")[i],time_stock,f,i+"eco_forecast_of_forecast_level"))
