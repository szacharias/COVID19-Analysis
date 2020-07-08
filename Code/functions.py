import pandas
from matplotlib import pyplot as plt
import seaborn
import sklearn
from sklearn.metrics import mean_squared_error
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from math import sqrt
from dash.dependencies import Input, Output 
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.statespace.sarimax import SARIMAX

register_matplotlib_converters()

def df_to_timeseries(single_country, period ):
    
    # Process raw(ungrouped) data
#     instance = single_country.drop(["Province/State","Country/Region","Lat","Long"],axis = 1).transpose()

    # Process grouped data
    instance = single_country.drop(["Country/Region","Lat","Long"],axis = 1).transpose()
    
    instance_array = np.asarray(instance)
    difference_list = []
#     for i in range(0 , len(instance)-1):
#         difference_list.append( int(instance[i+1]) - int(instance[i]) ) 
    difference_list = instance.diff(periods = period)
    difference_list = difference_list[1:].dropna().astype(int)
    
    instance_array = instance_array[1:].astype(int) 
    
    return instance_array, difference_list
 
def df_groupby_countries( df ):
    
    df_country = df.groupby("Country/Region")
    df_country = df_country.aggregate(np.sum)  
    df_country.reset_index(level=0, inplace=True) 
    
    return df_country


def plt_all_cases_increase_cases(country , dataset , period = 7 , figure_size=20 ):

    confirmed = dataset.loc[dataset["Country/Region"]==country].head()
    
    ## Format into timeseries data
    all_cases, increased_case = df_to_timeseries(confirmed, period)

    figure = plt.figure(figsize= (figure_size, figure_size/2 ))

    ## Total Infected Case
    plt.subplot(2,2,1)
    plt.plot(all_cases)
    plt.title("All Cases from " + str(country))

    ## Increase Case
    plt.subplot(2,2,2)
    plt.plot(increased_case)
    plt.title("Increased Cases from " + str(country))
    
    #figure.autofmt_xdate()
#     plt.savefig("img/" +  str(country) + "_all_w_increased_cases" +".png", bbox_inches='tight')
    plt.savefig("img/" +  str(country) + "_all_w_increased_cases" +".png")
    plt.close('all')


def plot_all( total_cases, increased_case, yhat, country = "Taiwan"):
### Plottinㄕ
#     predicted_cases = np.concatenate((total_cases[:,0], yhat + total_cases[-1,0]))
    predicted_cases = np.concatenate((total_cases[:,0],  yhat)) 
    figure = plt.figure(figsize=(10,10))

    plt.subplot(2,2,1)
    plt.title("Original Total Cases")
    plt.plot(total_cases)
    
    plt.subplot(2,2,2)
    plt.plot(increased_case.dropna())
    plt.title("Increased case")
    plt.xticks(rotation=45)
    plt.xticks(np.arange(0, len(increased_case.dropna()), (len(increased_case.dropna()) / 7))) 
    
    plt.subplot(2,2,3)
    plt.plot(predicted_cases)
    plt.title("Total Cases and forcasted")


    plt.subplot(2,2,4)
    plt.plot(yhat)
    plt.xticks(rotation=90)
    plt.title("yhat")
    
    plt.savefig("img/" + str(country)+ "_all_charts.png",  bbox_inches='tight')
    plt.close('all')
    
def SARIMA_PREDICT(cases, title , is_increase_case = False , order_tuple = (1,1,1), seasonal_tuple = (1,1,1,1), fit_param = (False, False )):
    initial_size = int(len(cases) * 1 / 3)
    train, test  =  cases[0:initial_size], cases[initial_size:len(cases)]
    history = [x for x in train]
    SARIMA_predictions = []
    
    
    model = SARIMAX(history, order = order_tuple )
    model_fit = model.fit(disp=False, transparams=False) 
    if not is_increase_case:    
        model = SARIMAX(history, order = order_tuple, seasonal_order=(1, 1, 1, 1))
        model_fit = model.fit(disp=fit_param[0], transparams=fit_param[1]) 
        
    for t in range(len(test)):
        
        model = SARIMAX(history, order = order_tuple )
        model_fit = model.fit(disp=False, transparams=False) 
        if not is_increase_case:    
            model = SARIMAX(history, order = order_tuple, seasonal_order=(1, 1, 1, 1))
            model_fit = model.fit(disp=fit_param[0], transparams=fit_param[1]) 
        
        yhat = model_fit.forecast()[0]
        SARIMA_predictions.append(yhat)
        history.append(test[t])
    rmse = sqrt(mean_squared_error(test, SARIMA_predictions)) / len(cases)
    print('Test RMSE: %.3f' % rmse)

    figure = plt.figure(figsize=(20,10))
    plt.title(title) 
    plt.plot(test)  
    plt.plot(SARIMA_predictions)
    
    plt.legend(['Actual' , 'Predicted']) 
    return SARIMA_predictions, model_fit


## this does not predict the past fourteen days for verification uses
def SARIMA_PREDICT_edit(cases, title , is_increase_case = False , order_tuple = (1,1,1), seasonal_tuple = (1,1,1,1), fit_param = (False, False )):
    initial_size = int(len(cases) * 1 / 3)
    train, test  =  cases[0:initial_size], cases[initial_size:len(cases)-14]
    history = [x for x in train]
    SARIMA_predictions = []
    
    test_length = len(test) 
    
    model = SARIMAX(history, order = order_tuple )
    model_fit = model.fit(disp=False, transparams=False) 
    if not is_increase_case:    
        model = SARIMAX(history, order = order_tuple, seasonal_order=(1, 1, 1, 1))
        model_fit = model.fit(disp=fit_param[0], transparams=fit_param[1]) 
        
    for t in range(test_length):
        
        model = SARIMAX(history, order = order_tuple )
        model_fit = model.fit(disp=False, transparams=False) 
        if not is_increase_case:    
            model = SARIMAX(history, order = order_tuple, seasonal_order=(1, 1, 1, 1))
            model_fit = model.fit(disp=fit_param[0], transparams=fit_param[1]) 
        
        yhat = model_fit.forecast()[0]
        SARIMA_predictions.append(yhat)
        history.append(test[t])
    rmse = sqrt(mean_squared_error(test, SARIMA_predictions)) / test_length
    print('Test RMSE: %.3f' % rmse)

    figure = plt.figure(figsize=(20,10))
    plt.title(title) 
    plt.plot(test)  
    plt.plot(SARIMA_predictions)
    
    plt.legend(['Actual' , 'Predicted']) 
    return SARIMA_predictions, model_fit