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


rp = rz.Plotting(False)
Frequency = ("m","d")
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

    if f == "d":
        df_equity_ret_squared = df_equity_ret ** 2

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
    check = 4
    while check <= 5:
        if check == 4:
            
            #delete last 24 month and delete null value or delete 5% of daily
            n = int(np.round(df_equity_L.dropna().shape[0] * 0.05) ) if f == "d" else 24
            nlag = 22 if f == "d" else 12
            #Cuts
            df_eco_cut = Economic_Data.iloc[:-24,:]
            df_eco_ret_cut = df_eco_ret.dropna().iloc[:-24,:]
            df_log_cut = df_equity_L.dropna().iloc[:-n,:]
            df_ret_cut = df_equity_ret.dropna().iloc[:-n,:]
            
            if f == "d":
                df_equity_ret_squared_cut = df_equity_ret_squared.dropna().iloc[:-n,:]
            
            nnlg = 20

        else:
            #Taking the last two years and last six month
            shape=df_equity_L.dropna().shape[0]
            
            n = int(np.round(shape * 0.05) ) if f == "d" else 24
            nlag = 8 if f == "d" else 4
            
            #Cuts
            
            df_eco_cut = Economic_Data.tail(24)
            df_eco_ret_cut = df_eco_ret.dropna().tail(24)
            df_log_cut = df_equity_L.dropna().tail(n)
            df_ret_cut = df_equity_ret.dropna().tail(n)
            if f == "d":
                df_equity_ret_squared_cut = df_equity_ret_squared.dropna().tail(n)
            
            nnlg = 4

        print(check)
        adf_log = rf.adf_test(df_log_cut,nlag)
        adf_ret=rf.adf_test(df_ret_cut,nlag)
        adf_eco=rf.adf_test(df_eco_cut,nlag)
        adf_eco_ret_cut=rf.adf_test(df_eco_ret_cut,nlag)

        adf_log.to_excel(rz.folder_definer(str(check)+"excel")+"adf_log_"+f+".xlsx")
        adf_ret.to_excel(rz.folder_definer(str(check)+"excel")+"adf_ret_"+f+".xlsx")
        adf_eco.to_excel(rz.folder_definer(str(check)+"excel")+"adf_eco.xlsx")
        adf_eco_ret_cut.to_excel(rz.folder_definer(str(check)+"excel")+"adf_eco_ret_cut.xlsx")
        
        #to excel, eco and log
        #SE sono stazionari fare arma altrimenti renderli stazionari
        ret_stat,ret_non_stat=rf.stationarity_and_not_stationary(adf_log)
        eco_stat,eco_non_stat=rf.stationarity_and_not_stationary(adf_eco)
        eco_ret_stat,eco_ret_non_stat=rf.stationarity_and_not_stationary(adf_eco_ret_cut)
     

        for i in adf_log.columns:
            rp.plotbar(adf_log[i],f,str(check)+"Log") 
        for i in adf_eco.columns:
            rp.plotbar(adf_eco[i],f,str(check)+"Log")
        

        #First difference
        df_ret_cut = df_ret_cut.loc[:,ret_non_stat]

        for i in ret_stat:
            df_ret_cut[i] = df_log_cut.iloc[:-n,:][i]
            if check == 5:
                df_ret_cut[i] = df_log_cut[i]

        df_eco_ret_cut = df_eco_ret_cut.loc[:,eco_ret_non_stat]

        for i in eco_ret_stat:
            df_eco_ret_cut[i] = df_eco_cut[i]
        
        
        adf_ret_cut=rf.adf_test(df_ret_cut.dropna(),nlag)

        adf_eco_ret_cut=rf.adf_test(df_eco_ret_cut.dropna(),nlag)
        
        
        
        adf_ret_cut.to_excel(rz.folder_definer(str(check)+"excel_first_diff")+"adf_ret_cut_"+f+"_post_first_diff.xlsx")
        adf_eco_ret_cut.to_excel(rz.folder_definer(str(check)+"excel_first_diff")+"adf_eco_ret_post_first_diff.xlsx")


        for i,y in zip(adf_ret_cut.columns,adf_eco_ret_cut.columns):
            rp.plotbar(adf_ret_cut[i],f,str(check)+"ret_first_diff")
            rp.plotbar(adf_eco_ret_cut[y],f,str(check)+"eco_ret_first_diff")
        
        rp.histo_plot2(df_log_cut,df_ret_cut,f)
        rp.histo_plot2(df_eco_cut,df_eco_ret_cut,f,90)

        arma_ret = rf.arma(df_ret_cut.dropna(how="all"), str.upper(f), 
                                maxlag = 3,
                                criterion = "AIC",n_steps=1)
        arma_eco = rf.arma(df_eco_ret_cut.dropna(how="all"), "M", 
                                maxlag = 3,
                                criterion = "AIC",n_steps=1)
        
        
        arma_ret.to_excel(rz.folder_definer(str(check)+"excel_arma")+"arma_ret"+f+".xlsx")
        arma_eco.to_excel(rz.folder_definer(str(check)+"excel_arma")+"arma_eco.xlsx")
        
        rp.resid_graph(arma_ret,f+"_"+str(check), nlg = nnlg)

        rp.resid_graph(arma_eco, "Monthly"+"_"+str(check), nlg = nnlg)

        
        rp.plot_correlation(arma_ret["resid"], 20, str(check)+"arma"+f,"resid_arma_ret")
        rp.plot_correlation(arma_eco["resid"], 20, str(check)+"arma","resid_arma_eco") 
        
        j_ret=rf.jungbox_test(arma_ret["resid"],10)
        j_eco=rf.jungbox_test(arma_eco["resid"],10)
        j_ret.to_excel(rz.folder_definer(str(check)+"excel_jbox")+"ret_jbox"+f+".xlsx")
        j_eco.to_excel(rz.folder_definer(str(check)+"excel_jbox")+"eco_jbox"+f+".xlsx")



        if check == 5:
        
            lcol = [ "forecast_value","true_value","finterval"]
            df_ret_to_print = pd.DataFrame(index = df_ret_cut.columns, columns = lcol)
            
            for i in arma_ret.index:
                l = []
                print(arma_ret.loc[i]["fvalue"])
                l.append(arma_ret.loc[i]["fvalue"])
                l.append(df_ret_cut[i].tail(1))
                l.append(arma_ret.loc[i,"finterval"])
                df_ret_to_print.loc[i,:] = l
            df_ret_to_print.to_excel("arma_ret"+f+".xlsx")


            df_eco_to_print = pd.DataFrame(index = df_eco_ret_cut.columns, columns = lcol)
            
            for i in arma_eco.index:
                l = []
                print(arma_eco.loc[i]["fvalue"])
                l.append(arma_eco.loc[i]["fvalue"])
                l.append(df_eco_ret_cut[i].tail(1))
                l.append(arma_eco.loc[i,"finterval"])
                df_eco_to_print.loc[i,:] = l
            df_eco_to_print.to_excel("arma_eco.xlsx")
            




        """
        POINT SEVEN
        """



        if f == "d":
            adf_equity_ret_squared_cut = rf.adf_test(df_equity_ret_squared_cut,nlag)


            adf_equity_ret_squared_cut.to_excel(rz.folder_definer(str(check)+"excel")+"adf_equity_ret_squared_cut.xlsx")


            squared_stat,squared_non_stat=rf.stationarity_and_not_stationary(adf_equity_ret_squared_cut)


            for i in adf_equity_ret_squared_cut.columns:
                rp.plotbar(adf_equity_ret_squared_cut[i],f,str(check)+"squared_ret") 
            

            df_equity_ret_squared_cut = df_equity_ret_squared_cut.loc[:,squared_non_stat]

            for i in squared_stat:
                df_equity_ret_squared_cut[i] = df_log_cut.iloc[:-n,:][i]
                if check == 5:
                    df_equity_ret_squared_cut[i] = df_log_cut[i] 

            adf_equity_ret_squared_cut=rf.adf_test(df_equity_ret_squared_cut.dropna(),nlag)
            adf_equity_ret_squared_cut.to_excel(rz.folder_definer(str(check)+"excel_first_diff")+"adf_equity_ret_squared_cut.xlsx")

            for i in adf_equity_ret_squared_cut.columns:
                rp.plotbar(adf_equity_ret_squared_cut[i],f,str(check)+"adf_equity_ret_squared_cut")
            rp.histo_plot2(df_log_cut,adf_equity_ret_squared_cut,f)

            arma_ret_squared = rf.arma(df_equity_ret_squared_cut.dropna(how="all"), "M", 
                                maxlag = 3,
                                criterion = "AIC",n_steps=1)
            arma_ret_squared.to_excel(rz.folder_definer(str(check)+"excel_arma")+"arma_ret_squared"+f+".xlsx")
            rp.resid_graph(arma_ret_squared,f+"_squared_"+str(check), nlg = nnlg)
            rp.plot_correlation(arma_ret_squared["resid"], 20, str(check)+"arma","arma_ret_squared")
            j_ret_squared=rf.jungbox_test(arma_ret_squared["resid"],10)
            j_ret_squared.to_excel(rz.folder_definer(str(check)+"excel_jbox")+"j_ret_squared"+f+".xlsx")

            if check == 5:
                lcol = [ "forecast_value","true_value","finterval"]
                df_ret_squared_to_print = pd.DataFrame(index = df_equity_ret_squared_cut.columns, columns = lcol)
                
                for i in arma_ret_squared.index:
                    l = []
                    print(arma_ret_squared.loc[i]["fvalue"])
                    l.append(arma_ret_squared.loc[i]["fvalue"])
                    l.append(df_equity_ret_squared_cut[i].tail(1))
                    l.append(arma_ret_squared.loc[i,"finterval"])
                    df_ret_squared_to_print.loc[i,:] = l
                df_ret_squared_to_print.to_excel("arma_squared"+f+".xlsx")

        check = check+1




        
