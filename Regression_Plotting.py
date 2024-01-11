from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import warnings
import statsmodels.graphics.tsaplots as smp
from Regression_function import sp
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns

global allow_clean
allow_clean = False
if 'SPY_PYTHONPATH' in os.environ:
    allow_clean = True
    """
    Check Last line for clean variables
    """

"""Pre_config END"""
        
def folder_definer(folder):
        cwd = os.getcwd()
        PATH = cwd + "/"+folder+"/"

        if not os.path.exists(PATH):
            os.mkdir(PATH)
        
        return PATH
def controlla_permesso(f):
    def wrapper(self, *args, **kwargs):
        if self.allow_execution:
            return f(self, *args, **kwargs)
        else:
            return None
    return wrapper

class Plotting:
    def __init__(self, allow_execution):
        self.allow_execution = allow_execution

    def test(self,table_str):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.1, 0.1, table_str, fontsize=10, va='center', ha='left')
        ax.axis('off')
        plt.show()

    @controlla_permesso
    def plot_correlation(self,Stocks,nlg,f,diff):    
        for i in Stocks.columns:
            smp.plot_acf(Stocks.loc[:,i], lags = nlg)
            plt . title ('Log prices - '+f+' - ACF ')
            plt.savefig(folder_definer(f+"_Correlation_ACF")+"/"+i+"_Correlation_ACF.png", dpi = 300)

            smp.plot_pacf(Stocks.loc[:,i], lags = nlg)
            plt . title ('Log prices - '+f+' - PACF ')
            plt.savefig(folder_definer(f+"__Correlation_PACF")+"/"+i+"-"+diff+"_Correlation_PACF.png", dpi = 300)

            plt.close("all")

    @controlla_permesso
    def plot_simple(self,stocks,frequency):
         for i in stocks.columns:
             plt.plot(stocks.loc[:,i])
             plt . xlabel ('Time - '+frequency+' - 30/09/2013 - 30/09/2023 ')
             plt . ylabel (i)
             plt.savefig(folder_definer(frequency+"_LogLevel")+"/"+i+"_LogLevel.png", dpi = 300)
             plt.close()
    @controlla_permesso
    def plot_line(self,stocks,time_series,frequency):
            for i in stocks.columns:
                plt . plot (time_series ,stocks.loc[:,i])
                plt . xlabel ('Time - '+frequency+' - 30/09/2013 - 30/09/2023 ')
                plt . ylabel (i)
                plt.savefig(folder_definer(frequency+"_LogLevel")+"/"+i+"_LogLevel.png", dpi = 300)
                plt.close()


    @controlla_permesso
    def plot_line2(self,stocks,time_series,frequency):
            for i in stocks.columns:
                plt . plot (time_series ,stocks.loc[:,i])
                plt . xlabel ('Time - '+frequency+' - 30/09/2013 - 30/09/2023 ')
                plt . ylabel (i)
                plt.savefig(folder_definer(frequency+"_LogPrice")+"/"+i+"_LogPrice.png", dpi = 300)
                plt.close()
    def plot_line3(self,stocks,frequency):
            for i in stocks.columns:
                m=stocks[i] . mean ()
                s=stocks[i] . std ()
                ax1 = plt . subplot (221)
                plt . plot ( stocks[i] )
                ax1 . set_title (i)
                ax2 = plt . subplot (222)
                plt . plot ( np . random . randn (stocks.shape[0] ,1) *s+ m)
                ax3 = plt . subplot (223)
                plt.savefig(folder_definer(frequency+"_Random")+"/"+i+"_Random.png", dpi = 300)
                plt.close()
    def prob_plot(self,stock,frequency):
            for i in stock.columns:
                ax1 = plt . subplot ()
                sp . stats . probplot ( stock[i], dist ="norm",plot = plt )    
                ax1 . set_title (i)
                ax1 . set_xlabel ("")
                plt.savefig(folder_definer(frequency+"_Prob")+"/"+i+"_LogProb .png", dpi = 300)
                plt.close()
    @controlla_permesso
    def histo_plot(self,stock,frequency,bins = None):
        if bins is None: 
            if frequency == "d":
                bins = 1000
            else:
                bins = 90
        for i in stock.columns:
                ax1 = plt . subplot ()
                plt . hist ( np.array(stock['ADF']), bins)    
                ax1 . set_title (i)
                ax1 . set_xlabel ("")
                plt.savefig(folder_definer(frequency+"_histo")+"/"+i+"_Loghisto .png", dpi = 300)
                plt.close()
    @controlla_permesso
    def histo_plot2(self,stock,stock2,frequency,bins = None):
        if bins is None: 
            if frequency == "d":
                bins = 1000
            else:
                bins = 90
        for i in stock.columns:
                ax1 = plt . subplot ()
                plt . hist ( stock[i], bins,density=True)
                plt.title(i)
                plt.savefig(folder_definer(frequency+"_histo")+"/"+i+"_Loghisto .png", dpi = 300)
                plt.figure()
                plt . hist ( stock2[i], bins,density=True)   
                plt.title(i)
                plt.savefig(folder_definer(frequency+"_histo")+"/"+i+"_rethisto .png", dpi = 300)
                plt.close("all")

    
    def resid_graph(self,df,f,nlg):
        
        for i in df.index:
            
            smp.plot_acf(df.loc[i, "resid"], lags = nlg)
            plt . title ('Residual- '+f+' - ACF - '+ "var")
            plt.savefig(folder_definer(f+"__Correlation_ACF_resid_arma")+"/"+i+"_Correlation_ACF.png", dpi = 300)
                    
        
            smp.plot_pacf(df.loc[i, "resid"], lags = nlg)
            plt . title ('Log prices - '+f+' - PACF ')
            plt.savefig(folder_definer(f+"__Correlation_PACF_resid_arma")+"/"+i+"_Correlation_PACF.png", dpi = 300)
            plt.close("all")
        
    def plot_forecast(self,df, df_ret, time,f,name):
        x=  pd.date_range(time, periods=len(df_ret), freq=str.upper(f)) 
      
        if len(x)>len(df):
            return None
        # add the dates and the data to a new dataframe
        df_ret = pd.DataFrame({'dates': x, 'data': df_ret})
        # set the dataframe index to be the dates column
        df_ret = df_ret.set_index('dates')

        # Plot each line with a label
        plt.plot(x, df["Prediction"], label="Prediction")
        plt.plot(x, df["Lower_Bound"], label="Lower_Bound")
        plt.plot(x, df["Upper_Bound"], label="Upper_Bound")
        plt.plot(x, df_ret, label="Actual Data")

        plt.savefig(folder_definer(f+"_Forecast")+"/"+name+"_forecast.png", dpi = 300)
        plt.close("all")
    
    def plot_for_random(self,frw,rw,df,f,time):

        x=  pd.date_range(time, periods=len(rw), freq=str.upper(f))
        # add the dates and the data to a new dataframe
        rw = pd.DataFrame({'dates': x, 'data': rw})
        # set the dataframe index to be the dates column
        rw = rw.set_index('dates')

        frw = pd.DataFrame({'dates': x, 'data': frw})
        # set the dataframe index to be the dates column
        frw = frw.set_index('dates')

        df = pd.DataFrame({'dates': x, 'data': df})
        # set the dataframe index to be the dates column
        df = df.set_index('dates')

        # Plot each line with a label

        plt.plot(x, df["data"]["Prediction"], label="Prediction")
        plt.plot(x, df["data"]["Lower_Bound"], label="Lower_Bound")
        plt.plot(x, df["data"]["Upper_Bound"], label="Upper_Bound")
        plt.plot(x, df["data"]["true_value"], label="Actual Data")
        plt.plot(x, frw["data"]["Prediction"], label="Random Walk Forecast")
        plt.plot(x, rw["data"], label="Random Walk value")
        plt.show()
        plt.close("all")

    def plot_ljung_box(self,df, f,check):
            
        for i in df.index:
        
            Plotting(True).plotbar(df.loc[i,:], f,diff=str(check)+i+"jingle_box")




    """
    
    ############################## OLD PROJECT #####################
    """
    
    def plotbar(self,P,frequency,diff, one_value = 0.01, five_value = 0.05, 
                ten_value = 0.1, obj = '',):
        """/3_p_value_plots/"""
        variable = P.name
        P = pd.DataFrame(data = P, columns = [variable])
        try:
            mean = P.loc['Mean', variable]
        except:
            mean= 0.0
        if obj == '':
            obj = variable
        P['stock_names'] = P.index

        def bar_highlight(value, one_value, 
                        five_value,
                        ten_value,
                        mean):
            
            if value <= one_value:
                return 'red'
            elif value <= five_value:
                return 'orange'
            elif value <= ten_value:
                return 'gold'
            if value == mean:
                return 'black'
            else:
                return 'grey'
        fig, ax = plt.subplots()   
    
        P['colors'] = P[variable].apply(bar_highlight, args = (one_value, 
                            five_value,
                            ten_value,
                            mean))

        bars = plt.bar(P['stock_names'], P[variable], color=P['colors'])
        x_pos = range(P['stock_names'].shape[0])
        plt.xticks(x_pos, P['stock_names'], rotation=90)
        plt.title(obj)
        variable = variable.replace(":","_")
        plt.savefig(folder_definer(frequency+"_plot")+"/"+variable+"_"+diff+".png")
        plt.close()
        
    @controlla_permesso
    def plotbar_DW(self,P,SavePath, Lbound = 0.01, Ubound= 0.05, 
                   obj = '', conf = 0.05,
                   pos_autocorr = True):
        """/3_p_value_plots/"""
        variable = P.name
        P = pd.DataFrame(data = P, columns = [variable])
        mean = P.loc['Mean', variable]
        P['stock_names'] = P.index

        def bar_highlight(value, Lbound, Ubound, mean):
            
            if pos_autocorr:
                #there is statistical evidence that the error terms are positively autocorrelated
                if value <= Lbound:
                    return 'red'
                #there is no statistical evidence that the error terms are positively autocorrelated
                elif value >= Ubound:
                    return 'grey'
    
                if value == mean:
                    return 'black'
                #the test is inconclusive
                else:
                    return 'blue'
                
            else: 
                
                #there is statistical evidence that the error terms are negatively autocorrelated
                if (4 - value) <= Lbound:
                    return 'red'
                #there is no statistical evidence that the error terms are negatively autocorrelated
                elif (4-value) >= Ubound:
                    return 'grey'
                
                if value == mean:
                    return 'black'
                #the test is inconclusive
                else:
                    return 'blue'
                
        fig, ax = plt.subplots()   
    
        P['colors'] = P[variable].apply(bar_highlight, args = (Lbound, Ubound, mean))

        bars = plt.bar(P['stock_names'], P[variable], color=P['colors'])
        x_pos = range(P['stock_names'].shape[0])
        plt.xticks(x_pos, P['stock_names'], rotation=90)
        if pos_autocorr:
            plt.title("{name}: H0 = Absence of positive autocorrelation, confidence level = {cf}".format(
                name = obj, cf = conf))
            
        else:
            plt.title("{name}: H0 = Absence of negative autocorrelation, confidence level = {cf}".format(
                name = obj, cf = conf))
        
        variable = variable.replace(":","_")
        plt.savefig(folder_definer(SavePath)+"/"+variable+".png")
        if allow_clean:
            plt.show()
        
        plt.close()

    @controlla_permesso
    def plotCAPM(self,rStock,Market,stock_names,OLSRes,SavePath):
        """/2_testCAPM/"""
        for e in stock_names:

            plt.figure()
            plt.plot(Market, OLSRes.loc[e, 'beta: Market']*Market+OLSRes.loc[e, 'Alpha'])
            plt.scatter(Market,rStock[e])
            plt . xlabel ('Eurostoxx')
            plt . ylabel (e)
            plt.savefig(folder_definer(SavePath[0])+"/"+SavePath[1]+"-"+e+SavePath[2]+".png")
            plt.close()

    """
    - setx = market return
    - set y = stock returns
    - title = title of the graph
    - xlabel = name of the x-axis
    - ylabel = name of the y-axis
    - sigla = unique name (either ER for euribor or ERB for bunds) to differentiate
            between plots of excess returns of stocks computed with Euribor
            or with the bund yield.
    - Subset = dataframe with the data of total index returns
    - string_to_save = name of the folder in which the graph will be saved in the 
                    working directory
    """
    @controlla_permesso
    def plotscatter(self,setx,sety,title,xlabel,ylabel,sigla,SavePath):  

        l = pd.DataFrame(index = sety.columns, columns= ['Plot'])
            
        for e in sety.columns:
            
            ax1 = plt.figure().add_subplot()
            
            ax1.scatter(setx,sety[e])
            plt.title(title)
            plt . xlabel (xlabel)
            plt . ylabel (e+ylabel)
            plt.savefig(folder_definer(SavePath[0])+"/"+SavePath[1]+"-"+e+"_"+sigla+".png")
            l.loc[e,'Plot'] = plt.figure()
            plt.close("all")
            
        return l
    @controlla_permesso
    def comparison_scatter(self,df_stocks, df_portfolios, market,
                        title,xlabel,ylabel,
                        SavePath,CAPM_Port=None):
        ax1 = plt.figure().add_subplot()
        
        for i in df_stocks.columns:
            ax1.scatter(market, df_stocks.loc[:,i], 
                        c = 'silver', alpha = 0.5)
        plt.title(title)
        plt . xlabel (xlabel)
        plt . ylabel (ylabel)
        
        ax1.scatter(market, df_portfolios, c = 'black')

        if CAPM_Port is not None:
            ax1.plot(market, CAPM_Port.loc['Portfolio - EW','beta: Market']*
                    market+CAPM_Port.loc['Portfolio - EW','Alpha'])
            
        plt.savefig(folder_definer(SavePath[0])+"/"+SavePath[1]+"-"+i+".png")
        if allow_clean:
            plt.show()
        plt.close()

    @controlla_permesso
    def m_scatter(self,CAPM_summary, df_factors, df_stocks,
                SavePath):
        """
        /2_scatter_comparison/
        """
        
        x = list(CAPM_summary.index)
        y = x[:3] + x[-1:-4:-1]

        figure, axis = plt.subplots(2, 3) 

        for i in range(3):

            axis[0,i].scatter(df_factors['Market'],
                                    df_stocks.loc[:,y[i]])

        for i in range(3,6):
            axis[1,i-3].scatter(df_factors['Market'],
                                    df_stocks.loc[:,y[i]]) 
        
        plt.savefig(folder_definer(SavePath[0])+"/"+SavePath[1]+".png")

        if allow_clean:
            plt.show()
        plt.close()
    @controlla_permesso  
    def comparison_barplot(self,FF_summary, CAPM_summary,
                           label1 = 'CAPM',
                           label2 = 'Fama-French',
                           legend_pos = 'best'):
        index = 1
        diction = {}
        for e,i in zip(range(1,FF_summary.shape[1]+1),list(FF_summary.columns)):
            diction.update({e: i})

        print('The possible comparisons are: {}\n'.format(diction.items()))
        index = int(input('Which one would you like to compare?(0 to stop)\n'))
        if index == 0 or index>(FF_summary.shape[1]+1):
            print("ENDED")
            return
        name=diction[index]
        
        index = np.arange(FF_summary.index.shape[0])
        bar_width = 0.35
        
        fig, ax = plt.subplots()
        summer = ax.bar(index, FF_summary.loc[:,name], bar_width,
                        label= label2)
        
        winter = ax.bar(index + bar_width, CAPM_summary.loc[:,name], bar_width,
                        label = label1)
        
        ax.set_xlabel('Company')
        ax.set_ylabel('Value')
        ax.set_title('Comparison between {CAPM} and {FF}: {n}'.format(n = name, CAPM = label1,
                                                                                  FF = label2))
        
        x_pos = range(FF_summary.index.shape[0])
        plt.xticks(x_pos, FF_summary.index, rotation=90)
        
        ax.set_xticklabels(FF_summary.index)
        ax.legend(loc = legend_pos)
        
        if allow_clean:
            plt.show()
        plt.close()
        
    @controlla_permesso
    def factor_plot(self,GETS_ad_hoc_summary,df_factors):
        
        l = [i for i in GETS_ad_hoc_summary.columns if i[0:4] == 'beta']
        
        df_to_plot = pd.DataFrame(columns = l, index = GETS_ad_hoc_summary.index)
        
        for i in l:
            
            k = np.array(GETS_ad_hoc_summary[i])
            
            for j in range(len(k)):
        
                if np.isnan(k[j]):
                    
                    a = 0 
                    #a = np.dtype('int64').type(a)
                    k[j] = a
                
                else:
                    
                    a = 1
                    #a = np.dtype('int64').type(a)
                    k[j] = a
                    
            df_to_plot[i] = k
        
        df_to_plot = df_to_plot.astype(int)
        df_to_plot.columns = df_factors.columns
        

        plt.figure(figsize=(8, 6))
        sns.heatmap(df_to_plot, cmap='Blues', fmt='d', cbar=False,
                    linewidths=0.5, linecolor='black')
        
        plt.xlabel('Factors')
        plt.ylabel('Stocks')
        
        plt.show()
        
        
        
    
    @controlla_permesso
    def plot_looking(self,time_series,df_factors,df_portfolios):
        """
        Check why the only relevant variable appears to be the market
        """

        plt.figure()

        plt.plot(time_series, df_factors.loc[:, 'Market'], label = 'Market')
        plt.plot(time_series, df_portfolios.iloc[:,0], label = 'Portfolio')
        plt.legend()
        plt.show()
        plt.close()  

    @controlla_permesso
    def fama_french_plotting(self,df, model):
        # Convert datetime objects to numerical values for plotting
        x_values = date2num(df['Date'])

        plt.figure()

        # Set up the bar plot
        plt.bar(x_values, height = 1, width = 25)


        plt.xticks(df['Date'], df['Date'])
        plt.xticks(rotation=90, ha='right')
        plt.title('Break date distribution ({})'.format(model))

        # Format the x-axis as dates


        # Display the plot
        plt.show()
    @controlla_permesso
    def chow_test_plotting(self,p_val_df, model):
        for i in p_val_df.columns:
        
            plt.figure()
            plt.plot(p_val_df.loc[:, i])
            plt.axhline(y= 0.05, color = 'red')
            plt.title(i+ " ({})".format(model))

            plt.show()




    @controlla_permesso    
    def plotting_CAPM_7(self, list_to_plot,d3,df_bd_CAPM_2,l_conf, end):

           
            for m in l_conf:

                for i in list_to_plot:
                    
                    l = df_bd_CAPM_2.loc[i]
                
                        
                    plt.figure()
                    
                    plt.plot(d3[i].index, d3[i][m+ '_UBound'], label = 'Upper bound')
                    
                    if m != 'Alpha':
                    
                        plt.plot(d3[i].index, d3[i]['beta: ' + m], label = 'Beta value')
                        
                    else:
                        
                        plt.plot(d3[i].index, d3[i][m], label = 'Alpha value')
                    
                    plt.plot(d3[i].index, d3[i][m+'_LBound'], label = 'Lower bound')
                    
                    if l.ndim > 1:
                        
                        for j in l['Date']:
                            
                            plt.axvline(j, color = 'red', linestyle = '--',
                                        label = 'Break date')
                    
                    else:
                        
                        for j in l:
                
                            plt.axvline(j, color = 'red', linestyle = '--',
                                        label = 'Break date')
                            
                    if m != 'Alpha':        
                        
                        plt.title(i +": {par} ({mo} months rolling window)".format(par = 'Value of beta '+ m, mo = end + 1))
                        
                    else:
                        
                        plt.title(i +": {par} ({mo} months rolling window)".format(par = 'Value of alpha', mo = end + 1))
                    
                    plt.legend()
                    
                    plt.show()

            """
            Plot of values that do not have a confidence interval
            """

            l_roll = ['R-Squared']

            for m in l_roll:

                for i in list_to_plot:
                    
                    l = df_bd_CAPM_2.loc[i]
                
                        
                    plt.figure()
                    
                    plt.plot(d3[i].index, d3[i][m], label = m)
                        
                
                    
                    if l.ndim > 1:
                        
                        for j in l['Date']:
                            
                            plt.axvline(j, color = 'red', linestyle = '--',
                                        label = 'Break date')
                    
                    else:
                        
                        for j in l:
                
                            plt.axvline(j, color = 'red', linestyle = '--',
                                        label = 'Break date')
                            
                        
                    plt.title(i +": {par} ({mo} months rolling window)".format(par = 'Value of '+m, mo =  end + 1))
                    
                    plt.legend()
                    
                    plt.show()
    
    
    def plotEstimator(self,df_to_plot):
        plt.plot()
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_to_plot, cmap='Blues', fmt='d', cbar=False,
                linewidths=0.5, linecolor='black')

        plt.xlabel('Factors')
        plt.ylabel('Stocks')


    plt.show()
    @controlla_permesso
    def histoplotting(self,resid_autocorr_CAPM,resid_autocorr_FF,n):
        
        for i in range(resid_autocorr_CAPM.shape[0]):

            plt.figure()
            
            plt.hist(resid_autocorr_FF.iloc[i,-1], label = 'Fama-French', bins = n, alpha = 0.5)
            
            plt.hist(resid_autocorr_CAPM.iloc[i,-1], label = 'CAPM', bins = n, alpha = 0.5)
            plt.legend()
            plt.title('Comparison between the distribution of the residuals: {}'.format(resid_autocorr_CAPM.index[i]))
            
            plt.show()
