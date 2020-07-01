import pandas
from matplotlib import pyplot as plt
import seaborn
import sklearn
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output 

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

# def df_to_timeseries_countries(country):
#     instance = np.asarray(single_country.drop(["Province/State","Country/Region","Lat","Long"],axis = 1).transpose()) 
#     difference_list = []
#     for i in range(0 , len(instance)-1):
#         difference_list.append( int(instance[i+1]) - int(instance[i]) ) 

#     return instance, difference_list
      
    
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
    plt.savefig("img/" +  str(country) + "_all_w_increased_cases" +".png", bbox_inches='tight')
    
    plt.close('all')


# cumulated_confirmed_country = cumulated_confirmed.groupby("Country/Region")
# cumulated_confirmed_country = cumulated_confirmed_country.aggregate(np.sum)  
# cumulated_confirmed_country.reset_index(level=0, inplace=True) 
# cumulated_confirmed_country