#!/usr/bin/env python
# coding: utf-8

# In[207]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import functions as covid_lib
from numpy import hstack, array
from random import random
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output 

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# ---
# ## 1st Dataset : John Hopkins Data

# In[208]:


## John Hopkins Center for Systems Science and Engineer Data Base URL
## Contains timeseries data
## This set of data is updated daily
baseURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"

cumulated_confirmed_url = baseURL + "time_series_covid19_confirmed_global.csv"
cumulated_deaths_url    = baseURL + "time_series_covid19_deaths_global.csv"
cumulated_recovered_url = baseURL + "time_series_covid19_recovered_global.csv"

cumulated_confirmed     = pd.read_csv(cumulated_confirmed_url)
cumulated_deaths        = pd.read_csv(cumulated_deaths_url)
cumulated_recovered     = pd.read_csv(cumulated_recovered_url)

## Processed Dataset
## Combine by Country/Region
cumulated_confirmed_country = covid_lib.df_groupby_countries(cumulated_confirmed)
cumulated_deaths_country = covid_lib.df_groupby_countries(cumulated_deaths)
cumulated_recovered_country = covid_lib.df_groupby_countries(cumulated_recovered)


# In[209]:


# US_confirmed.to_csv('cululated_confirmed.csv', index = False)


# In[210]:


print("Confirmed data shape : " + str(cumulated_confirmed_country.shape))
print("Recovered data shape : " + str(cumulated_deaths_country.shape))
print("Deaths data shape : " + str(cumulated_recovered_country.shape))


# In[211]:


# ## To find specific instances on certain countries
# ## Consider grouping by countries to simplify the analysis process
# cumulated_confirmed.loc[cumulated_confirmed["Country/Region"]=="US"].head()


# In[212]:


# cumulated_confirmed_country.head()


# In[213]:


unique_countries = cumulated_confirmed_country["Country/Region"].unique()


# In[214]:


cumulated_deaths_country[cumulated_deaths_country["Country/Region"]=="Taiwan*"]


# In[215]:


## Call this only if you wanna wait forever
proceed = False
if proceed:
    for country in unique_countries:
        covid_lib.plt_all_cases_increase_cases(country, cumulated_confirmed_country  )


# In[216]:


# US_confirmed = cumulated_confirmed.loc[cumulated_confirmed["Country/Region"]=="US"].head()
# US_confirmed

# all_cases, increased_case = covid_lib.df_to_timeseries(US_confirmed, 7)


# ### Taiwan Analysis

# In[217]:


# query_country = "Taiwan*"
# query_confirmed = cumulated_confirmed.loc[cumulated_confirmed["Country/Region"]==query_country]
# queried_total_cases, queried_increased_case = covid_lib.df_to_timeseries(query_confirmed, 1)
# model = SARIMAX(queried_total_cases, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
# model_fit = model.fit(disp=False)
# yhat = model_fit.predict(len(queried_total_cases), len(queried_total_cases)+10)
# print(yhat)
# covid_lib.plot_all( queried_total_cases, queried_increased_case, yhat , query_country )


# In[218]:


# TW_confirmed = cumulated_confirmed.loc[cumulated_confirmed["Country/Region"]=="Taiwan*"].head()
# total_cases, increased_case = covid_lib.df_to_timeseries(TW_confirmed, 7)


# ### US Analysis

# In[219]:


US_confirmed = cumulated_confirmed.loc[cumulated_confirmed["Country/Region"]=="US"].head()
total_cases, increased_case = covid_lib.df_to_timeseries(US_confirmed, 7)


# In[220]:


figure = plt.figure(figsize = (10,5))
plt.plot(increased_case)
plt.xticks(np.arange(0, len(increased_case.dropna()), 6)) 
plt.xticks(rotation=90)
plt.savefig("./img/Infected Case.jpg",  bbox_inches='tight')


# In[221]:


# covid_lib.plot_all( total_cases, increased_case, yhat , "US" )


# ---

# ### Full Model Building

# In[222]:



increased_case_21_train  = increased_case.iloc[:-21]
increased_case_14_train = increased_case.iloc[:-14]
increased_case_7_train  = increased_case.iloc[:-7]

increased_case_21_test = increased_case.iloc[-21:]
increased_case_14_test = increased_case.iloc[-14:]
increased_case_7_test = increased_case.iloc[-7:]


# In[223]:


## This only predicts for one day
predictions_total, model_total = covid_lib.SARIMA_PREDICT(total_cases,"Total Cases", order_tuple = (0,1,0), fit_param = (True, True))
# predictions_total, model_total = covid_lib.SARIMA_PREDICT_edit(total_cases,"Total Cases", order_tuple = (0,1,0), fit_param = (True, True))


# #### Predict future total Cases

# In[224]:

 
# # today and tomorrow

# rmse = sqrt(mean_squared_error(temp, total_cases[-14:])) / len(temp)
# print(rmse)


# temp = model_total.forecast(14)
# plt.plot(temp)
# plt.plot(total_cases[-14:])
# plt.title("This is what a SARIMA prediction looks like, RMSE : " + str(rmse))
# plt.savefig("./img/1st_SARIMA.png")




# ---

# #### Count Increase Model Building

# In[225]:


# Prediction for 14 days
increase_temp_14 = increased_case_14_train.values 
predictions_increase, model_increase = covid_lib.SARIMA_PREDICT(increase_temp_14, "Increase Cases",  is_increase_case = True, fit_param = (True,True) )


# In[226]:

prediction_14 = model_increase.forecast(14)
prediction_14 = prediction_14.reshape(14, 1)
overall_prediction_14 = np.append(increase_temp_14, prediction_14)
# temp_l = increase_temp_14.append(temp)

rmse = sqrt(mean_squared_error( increased_case_14_test.values, prediction_14)) / len(prediction_14)
rmse

fig = plt.figure(figsize=(10,10)) 
fig.tight_layout()
plt.subplots_adjust(hspace = 0.25)

plt.subplot(2,1,1)
plt.plot(overall_prediction_14)
plt.plot(increased_case)
plt.xticks(np.arange(0, len(overall_prediction_14), 7))  
plt.xticks(rotation = 40)
plt.legend(["Actual", "Prediction"])
plt.title("Overall : Prediction vs Actual ")

plt.subplot(2,1,2)
plt.plot(increased_case_14_test.values)
plt.plot(prediction_14)
plt.legend(["Actual", "Prediction"])
plt.title("Last 14 : Prediction vs Actual. RMSE : " + str(rmse))

plt.savefig("./img/SARIMA_NEXT_14_Prediction.png")


# In[227]:


# Prediction for 7 days
increase_temp_7 = increased_case_7_train.values 
predictions_increase, model_increase = covid_lib.SARIMA_PREDICT(increase_temp_7, "Increase Cases",  is_increase_case = True, fit_param = (True,True) )


# In[228]:


# plt.plot(increased_case)


# In[229]:


# plt.plot(predictions_increase)


# In[230]:

prediction_7 = model_increase.forecast(7)
prediction_7 = prediction_7.reshape(7, 1)
overall_prediction_7 = np.append(increase_temp_7, prediction_7)
# temp_l = increase_temp_14.append(temp)

rmse = sqrt(mean_squared_error( increased_case_7_test.values, prediction_7)) / len(prediction_7)
rmse

fig = plt.figure(figsize=(10,10))
fig.tight_layout()
plt.subplots_adjust(hspace = 0.25)
plt.subplot(2,1,1)
plt.plot(overall_prediction_7)
plt.plot(increased_case)
plt.xticks(np.arange(0, len(overall_prediction_7), 7))  
plt.xticks(rotation = 60)
plt.legend(["Actual", "Prediction"])
plt.title("Overall : Prediction vs Actual ")

plt.subplot(2,1,2)
plt.plot(increased_case_7_test.values)
plt.plot(prediction_7)
plt.legend(["Actual", "Prediction"])
plt.title("Last 7 : Prediction vs Actual. RMSE : " + str(rmse))
plt.savefig("./img/SARIMA_NEXT_14_Prediction.png")


# ---

# ### Using Sentiment Values

# In[231]:


def splitColumnSeries(sentiment_value_dataframe):
    
    SentimentValueTitle            = sentiment_value_dataframe.iloc[:,0]
    SentimentValueDescription      = sentiment_value_dataframe.iloc[:,1]
    SentimentValueTitleVader       = sentiment_value_dataframe.iloc[:,2]
    SentimentValueDescriptionVader = sentiment_value_dataframe.iloc[:,3]
    
    return (SentimentValueTitle, SentimentValueDescription ,SentimentValueTitleVader,SentimentValueDescriptionVader)


# In[232]:


sentiment_value_dataframe = pd.read_csv("../Data/SentimentValues.csv", index_col = [0] )
#  sentiment_value = sentiment_value_dataframe.iloc[:,-4:]
# sentiment_value_dataframe

#### Split into train test data
sentiment_value_dataframe_train_14 = sentiment_value_dataframe.iloc[:-14]
sentiment_value_dataframe_test_14 = sentiment_value_dataframe.iloc[-14:]

sentiment_value_dataframe_train_7 = sentiment_value_dataframe.iloc[:-7]
sentiment_value_dataframe_test_7 = sentiment_value_dataframe.iloc[-7:]


sentiment_value_dataframe_train_21 = sentiment_value_dataframe.iloc[:-21]
sentiment_value_dataframe_test_21 = sentiment_value_dataframe.iloc[-21:]


SentimentValueTitle            = sentiment_value_dataframe.iloc[:,0]
SentimentValueDescription      = sentiment_value_dataframe.iloc[:,1]
SentimentValueTitleVader       = sentiment_value_dataframe.iloc[:,2]
SentimentValueDescriptionVader = sentiment_value_dataframe.iloc[:,3]


# In[233]:


# #### 14 Days training data
# SentimentValueTitle, SentimentValueDescription ,SentimentValueTitleVader,SentimentValueDescriptionVader = splitColumnSeries(sentiment_value_dataframe_train_7)


# In[236]:


predicted_length = int(len(increased_case_7_train.values) * 1 / 3)
sarima_prediction = increased_case_7_train.iloc[0:predicted_length,0].append(pd.Series(predictions_increase), ignore_index = True)

sarima_prediction_seq_ = sarima_prediction.values.reshape(len(sarima_prediction),1)

# len(sarima_prediction_seq_)


# In[240]:


### JUST FOR REFERENCE
# increased_case_14_train = increased_case.iloc[:-14]
# increased_case_7_train  = increased_case.iloc[:-7]
# increased_case_14_test = increased_case.iloc[-14:]
# increased_case_7_test = increased_case.iloc[-7:]


def boosted_lstm(sentiment_value_dataframe , overall_prediction , target, testing_data, n_steps_in = 14, n_steps_out = 7):
    
    SentimentValueTitle, SentimentValueDescription ,SentimentValueTitleVader,SentimentValueDescriptionVader = splitColumnSeries(sentiment_value_dataframe)
    
    # define input sequence 
    
    predicted_length = int(len(overall_prediction.values) * 1 / 3)
    sarima_prediction = overall_prediction.iloc[0:predicted_length].append(pd.Series(predictions_increase), ignore_index = True)
    sarima_prediction_seq_ = sarima_prediction.values.reshape(len(sarima_prediction),1)
    out_seq = target.values.reshape(len(target.values), 1)
    
    SentimentValueDescription_seq     = SentimentValueDescription.values.reshape(len(SentimentValueDescription),1)
    SentimentValueTitleVader_seq      = SentimentValueTitleVader.values.reshape(len(SentimentValueTitleVader),1)
    SentimentValueDescriptionVader_seq= SentimentValueDescriptionVader.values.reshape(len(SentimentValueDescriptionVader),1)
    SentimentValueTitle_seq           = SentimentValueTitle.values.reshape(len(SentimentValueTitle),1)

    
    print("sarima_prediction_seq_" + str(len(sarima_prediction_seq_)))
# #     sarima_output_all = prediction.reshape(len(prediction),1)    
# #     print(len(sarima_output_all))
#     print(len(SentimentValueDescription_seq)  )
#     print(len(SentimentValueTitleVader_seq      )  )
#     print(len(SentimentValueDescriptionVader_seq)  )
#     print(len(SentimentValueTitle_seq           )  )
#     print(len(out_seq))
    
    
    # horizontally stack columns
    # dataset = hstack((in_seq1, in_seq2, out_seq))
    dataset = hstack((
                    sarima_prediction_seq_,
                    SentimentValueDescription_seq , 
                      SentimentValueTitleVader_seq, 
                      SentimentValueDescriptionVader_seq,
                      SentimentValueTitle_seq,
                      out_seq))

    ################### Building the testing dataset ###################
    
    ## Testing_Date : SENTIMENT_DATAFRAME, SARIMA_PREDICTION, TARGET_VALUE
    SentimentValueTitle_test, SentimentValueDescription_test ,SentimentValueTitleVader_test,SentimentValueDescriptionVader_test = splitColumnSeries(testing_data[0])
    
    SentimentValueDescription_seq_test     = SentimentValueDescription_test.values.reshape(len(SentimentValueDescription_test),1)
    SentimentValueTitleVader_seq_test      = SentimentValueTitleVader_test.values.reshape(len(SentimentValueTitleVader_test),1)
    SentimentValueDescriptionVader_seq_test= SentimentValueDescriptionVader_test.values.reshape(len(SentimentValueDescriptionVader_test),1)
    SentimentValueTitle_seq_test           = SentimentValueTitle_test.values.reshape(len(SentimentValueTitle_test),1)
    
#     print(len(testing_data[1]))
#     print(len(SentimentValueDescription_seq_test)  )
#     print(len(SentimentValueTitleVader_seq_test      )  )
#     print(len(SentimentValueDescriptionVader_seq_test)  )
#     print(len(SentimentValueTitle_seq_test           )  )
    
    test_input = hstack((
                    testing_data[1],
    SentimentValueDescription_seq_test     ,
    SentimentValueTitleVader_seq_test      ,
    SentimentValueDescriptionVader_seq_test,
    SentimentValueTitle_seq_test           ))
  


    ################### Multivariate, Multi Timestep LSTM ###################

    n_steps_in, n_steps_out = 14, 7
    X, y = covid_lib.split_sequences_three(dataset, n_steps_in, n_steps_out)
    n_features = X.shape[2] 
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
#     model.add(LSTM(100, activation='relu', return_sequences=True))
#     model.add(LSTM(100, activation='relu', return_sequences=True))

#     model.add(LSTM(100, activation='relu', return_sequences=True))

    model.add(LSTM(100, activation='relu'))

    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=200, verbose=0)
    
    
    ################### Testing Results ###################

#     x_input = test_input[-n_steps_in:]
    x_input = test_input    
    
    ### reshape : sample count, time step, features
    x_input = x_input.reshape((1, n_steps_in, n_features))
    # print(x_input)
    yhat = model.predict(x_input, verbose=1)
    print(yhat)
    
    
    return model,yhat

testing_data = (sentiment_value_dataframe_test_14, prediction_14 ,increased_case_14_test)
    
lstm_model,yhat = boosted_lstm(sentiment_value_dataframe_train_7 , 
                          sarima_prediction  ,
                          increased_case_7_train, 
                          testing_data)


# In[241]:



aggregate_df = increased_case.copy()
temp_sarima = sarima_prediction.copy()
temp_sarima
N = 4
for i in range(len(sarima_prediction) , len(aggregate_df)):
    temp_sarima = temp_sarima.set_value(i, np.nan)
sarima_df = pd.DataFrame({"SARIMA":temp_sarima, "Index": aggregate_df.index}).set_index(["Index"])
aggregate_df = pd.concat([aggregate_df, sarima_df], axis = 1)
# plt.plot(aggregate_df)


# In[242]:


full_sarima_prediction = temp_sarima.copy()
for i in range(189, 196):
    full_sarima_prediction.iloc[i] = prediction_7[i-189]


# In[243]:
sentiment_value_dataframe_test_21.index = increased_case_21_test.index
# sentiment_value_dataframe_test_21

def testing_boosted_lstm(sentiment_value_dataframe_test_21, sarima_prediction ,
                         target,full_sarima_prediction, lstm_model,
                    n_steps_in = 14, n_steps_out =  7 , n_features = 5):
    
    
    
    ## Testing_Date : SENTIMENT_DATAFRAME, SARIMA_PREDICTION, TARGET_VALUE
    SentimentValueTitle_test, SentimentValueDescription_test ,SentimentValueTitleVader_test,SentimentValueDescriptionVader_test = splitColumnSeries(sentiment_value_dataframe_test_21[:-7])
    
    SentimentValueDescription_seq_test     = SentimentValueDescription_test.values.reshape(len(SentimentValueDescription_test),1)
    SentimentValueTitleVader_seq_test      = SentimentValueTitleVader_test.values.reshape(len(SentimentValueTitleVader_test),1)
    SentimentValueDescriptionVader_seq_test= SentimentValueDescriptionVader_test.values.reshape(len(SentimentValueDescriptionVader_test),1)
    SentimentValueTitle_seq_test           = SentimentValueTitle_test.values.reshape(len(SentimentValueTitle_test),1)
    
    sarima_prediction = np.array(sarima_prediction).reshape(len(sarima_prediction),1)
    
#     print(len(sarima_prediction))
#     print(len(SentimentValueDescription_seq_test)  )
#     print(len(SentimentValueTitleVader_seq_test      )  )
#     print(len(SentimentValueDescriptionVader_seq_test)  )
#     print(len(SentimentValueTitle_seq_test           )  )
    
    test_input = hstack((
                    sarima_prediction,
    SentimentValueDescription_seq_test     ,
    SentimentValueTitleVader_seq_test      ,
    SentimentValueDescriptionVader_seq_test,
    SentimentValueTitle_seq_test           ))
    
    ################### Predict ###################

    
    x_input = test_input    
    
    ### reshape : sample count, time step, features
    x_input = x_input.reshape((1, n_steps_in, n_features))
    # print(x_input)
    yhat = lstm_model.predict(x_input, verbose=1)
    print(yhat)
#     plt.plot(target)
#     plt.plot(yhat[0].tolist())
#     plt.legend(["Actual","Predicted"])
#     plt.xticks(rotation = 90)

#     yhat_df = pd.DataFrame(np.vstack(yhat[0]), index = temp.index)



    yhat_df = pd.DataFrame(np.vstack(yhat[0]), index = increased_case_21_test[-7:].index)
    
    
    fig, ax = plt.subplots()
    fig.set_size_inches((20, 10))
    ax.plot(target.index,target, lw = 4, color = "salmon")
    ax.plot(target.index[-7:], yhat_df, lw = 4, color = 'cornflowerblue')
    ax.plot(target.index, full_sarima_prediction[-21:], lw= 4,color = "lightseagreen" )
    ax.tick_params(axis = "x", rotation = 45)
    ax.set_ylabel("Increased Count" , size = 14)
    ax.legend(["Target", "LSTM", "SARIMA"] ,loc = 2)
#     ax.xticks(rotation = 45)
    
    ax2 = ax.twinx()
    linestyles = '--'
    
    
    ax2.plot(sentiment_value_dataframe_test_21["SentimentValueTitleVader"], color = "brown", linestyle = linestyles)
    ax2.plot(sentiment_value_dataframe_test_21["SentimentValueDescriptionVader"], color = "grey", linestyle = linestyles)
    ax2.set_ylabel("Sentiment", size = 14)
    ax2.legend(["Title Sentiment", "Description Sentiment"],loc = 1)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    
#     fig = plt.figure(figsize=(20,15))
#     plt.subplots_adjust(hspace=0.25)
#     plt.subplot(2,1,1)
    
#     plt.plot(target, linewidth=3)
# #     plt.plot(sarima_prediction[-21:])  
#     plt.plot(yhat_df, lw = 3)
#     plt.plot(full_sarima_prediction[-21:].values, lw = 3)
#     plt.xticks(rotation = 45)
#     plt.legend(["Target", "SARIMA", "LSTM"])
#     plt.ylabel("Infected Count", size = 16)
#     plt.xlabel("Changes According to time", size = 16)
#     plt.grid("on")
    
#     plt.subplot(2,1,2)
#     plt.plot(sentiment_value_dataframe_test_21["SentimentValueTitleVader"])
#     plt.plot(sentiment_value_dataframe_test_21["SentimentValueDescriptionVader"])
# #     plt.plot(sentiment_value_dataframe_test_21.mean(axis=1).values)
#     plt.ylabel("Sentiment Value", size = 16)
#     plt.xlabel("Changes according to date" , size = 16)
#     plt.xticks(rotation = 45)
#     plt.grid("on")
#     sns.despine()
#     plt.savefig("FinalModelPrediction.png")
    
     ################### Validate ###################

    
    
    return lstm_model,yhat
  
    
# testing_data = testing_boosted_lstm(sentiment_value_dataframe_test_21, prediction_14 ,increased_case_21_test, lstm_model)


model, yhat = testing_boosted_lstm(sentiment_value_dataframe_test_21, sarima_prediction[-21:-7] ,increased_case_21_test, full_sarima_prediction,lstm_model)

# In[244]:


# def test_lstm( sentiment_value_dataframe , prediction , target, lstm_model):
  
#     SentimentValueTitle, SentimentValueDescription ,SentimentValueTitleVader,SentimentValueDescriptionVader = splitColumnSeries(sentiment_value_dataframe)
    
#     # define input sequence 
# #     prediction
# #     predicted_length = int(len(prediction) * 1 / 3)
# #     sarima_prediction = prediction[0:predicted_length].append(pd.Series(predictions_increase), ignore_index = True)
# #     sarima_prediction_seq_ = sarima_prediction.values.reshape(len(sarima_prediction),1)
#     out_seq = target.values.reshape(len(target.values), 1)
    
#     SentimentValueDescription_seq     = SentimentValueDescription.values.reshape(len(SentimentValueDescription),1)
#     SentimentValueTitleVader_seq      = SentimentValueTitleVader.values.reshape(len(SentimentValueTitleVader),1)
#     SentimentValueDescriptionVader_seq= SentimentValueDescriptionVader.values.reshape(len(SentimentValueDescriptionVader),1)
#     SentimentValueTitle_seq           = SentimentValueTitle.values.reshape(len(SentimentValueTitle),1)

    
    

#     test_input = hstack((
#                     prediction,
#                     SentimentValueDescription_seq , 
#                       SentimentValueTitleVader_seq, 
#                       SentimentValueDescriptionVader_seq,
#                       SentimentValueTitle_seq))


    
# # test_lstm(sentiment_value_dataframe_test_7 , prediction ,increased_case_7_test , lstm_model )


# ---

# In[ ]:


################### Correlation 


# In[285]:


temp_sentiment = sentiment_value_dataframe.copy()
temp_sentiment["IncreaseCount"] = increased_case.values
temp_sentiment


# In[292]:


corrMatrix = temp_sentiment.corr()

plt.figure(figsize=(10,10))
sns.heatmap( corrMatrix, annot=True)
# plt.figure(figsize = (20,20))
plt.savefig("./img/CorrelationMap.png")

# In[ ]:




