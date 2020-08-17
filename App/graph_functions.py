################### Graph Function for Dash


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import functions as covid_lib
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
from datetime import datetime
from os.path import isfile
import os
import plotly
import plotly.express as px
import graph_functions as gf
from plotly.subplots import make_subplots


baseLocation = "../Data/"

## Read from file
cumulated_confirmed_loc = baseLocation + "cumulated_confirmed.csv"
cumulated_deaths_loc    = baseLocation + "cumulated_deaths.csv"
cumulated_recovered_loc = baseLocation + "cumulated_recovered.csv"

cumulated_confirmed     = pd.read_csv(cumulated_confirmed_loc)
cumulated_deaths        = pd.read_csv(cumulated_deaths_loc)
cumulated_recovered     = pd.read_csv(cumulated_recovered_loc)

cumulated_confirmed_country = covid_lib.df_groupby_countries(cumulated_confirmed)
cumulated_deaths_country = covid_lib.df_groupby_countries(cumulated_deaths)
cumulated_recovered_country = covid_lib.df_groupby_countries(cumulated_recovered)

unique_countries = cumulated_confirmed_country["Country/Region"].unique()

period = 7

###################### Functions ######################

############################## Section 1 ##############################

def update_states(country):
    # d = allData() 
    d= cumulated_confirmed
    states = list(d.loc[d['Country/Region'] == country]['Province/State'].unique()) 
    states = [x for x in states if str(x) != 'nan']
    states.insert(0, '<all>') 
    state_options = [{'label':s, 'value':s} for s in states]
    state_value = state_options[0]['value'] 
    return state_options, state_value


def update_graph(country, state):  
    # print(state)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if state != '<all>':
        return_df_deaths = cumulated_deaths.loc[cumulated_deaths["Province/State"] == state]
        return_df_deaths = return_df_deaths.drop(["Province/State","Country/Region", "Lat", "Long" ],axis = 1)

        return_df_infected = cumulated_confirmed.loc[cumulated_confirmed["Province/State"] == state]
        return_df_infected = return_df_infected.drop(["Province/State","Country/Region", "Lat", "Long" ],axis = 1)

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x = return_df_deaths.columns, y=return_df_deaths.values[0], name="Deaths"),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(x= return_df_infected.columns, y=return_df_infected.values[0], name="Infected"),
            secondary_y=False,
        )

        # Add figure title
        fig.update_layout(
            title_text="COVID-19 Conditions in " + str(state) +"," +str(country)
        )

        # Set x-axis title
        fig.update_xaxes( dtick=7, nticks = 10,title_text='Timestamp') 

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Deaths</b>", secondary_y=True)
        fig.update_yaxes(title_text="<b>Infected</b>", secondary_y=False)

        return fig
    else :
        # fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        return_df_deaths = cumulated_deaths_country.loc[cumulated_deaths_country["Country/Region"]==country]
        return_df_deaths = return_df_deaths.drop(["Country/Region", "Lat", "Long" ],axis = 1)

        return_df_infected = cumulated_confirmed_country.loc[cumulated_confirmed_country["Country/Region"]==country]
        return_df_infected = return_df_infected.drop(["Country/Region", "Lat", "Long" ],axis = 1)

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x = return_df_deaths.columns, y=return_df_deaths.values[0], name="Deaths"),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(x= return_df_infected.columns, y=return_df_infected.values[0], name="Infected"),
            secondary_y=False,
        )

        # Add figure title
        fig.update_layout(
            title_text="COVID-19 Conditions in " + str(country)
        )

        # Set x-axis title
        fig.update_xaxes( dtick=7, nticks = 10,title_text='Timestamp') 

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Deaths</b>", secondary_y=True)
        fig.update_yaxes(title_text="<b>Infected</b>", secondary_y=False)

    return fig
        
def update_increase_graph(country, state): 
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if state != "<all>":
        infected_df = cumulated_confirmed.loc[cumulated_confirmed["Province/State"] == state]
        infected_df = infected_df.drop(["Province/State","Country/Region", "Lat", "Long"], axis = 1).diff(period , axis = 1).fillna(0)
        
        
        death_df = cumulated_deaths.loc[cumulated_deaths["Province/State"] == state]
        death_df = death_df.drop(["Province/State","Country/Region", "Lat", "Long"], axis = 1).diff(period, axis = 1).fillna(0)
        
        # Add traces
        fig.add_trace(
            go.Scatter(x = death_df.columns, y=death_df.values[0], name="Deaths"),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(x= infected_df.columns, y=infected_df.values[0], name="Infected"),
            secondary_y=False,
        )

        # Add figure title
        fig.update_layout(
            title_text="COVID-19 Conditions in " + str(state) +"," +str(country)
        )

        # Set x-axis title
        fig.update_xaxes(nticks = 10, title_text='Timestamp') 

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Deaths</b>", secondary_y=True)
        fig.update_yaxes(title_text="<b>Infected</b>", secondary_y=False)

        return fig
    else : 
        
        infected_df = cumulated_confirmed_country.loc[cumulated_confirmed_country["Country/Region"]==country]
        infected_df = infected_df.drop(["Country/Region", "Lat", "Long" ],axis = 1).diff(period, axis = 1).fillna(0)


        death_df = cumulated_deaths_country.loc[cumulated_deaths_country["Country/Region"]==country]
        death_df = death_df.drop(["Country/Region", "Lat", "Long" ],axis = 1).diff( period , axis = 1).fillna(0)

        # Add traces
        fig.add_trace(
            go.Scatter(x = death_df.columns, y=death_df.values[0], name="Deaths"),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(x= infected_df.columns, y=infected_df.values[0], name="Infected"),
            secondary_y=False,
        )

        # Add figure title
        fig.update_layout(
            title_text="COVID-19 Conditions in " + str(country)
        )

        # Set x-axis title
        fig.update_xaxes(nticks = 10,title_text='Timestamp') 

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Deaths</b>", secondary_y=True)
        fig.update_yaxes(title_text="<b>Infected</b>", secondary_y=False)

        return fig


############################## Section 2 ##############################
def news_display(news_date): 
    if news_date == "<all>":
        
        df = pd.read_csv("../Data/NewsContent.csv", index_col=[0])
        column_list = df.columns.drop(["SourceURL"])
        fig = go.Figure(data=[go.Table(
            columnorder = [1,2,3,4,5],
            columnwidth = [80,150,130,80,80],
            header=dict(values=list(column_list),
                        fill_color='paleturquoise' ),
            cells=dict(values=[df.Title, df.Description, df.NewsURL, df.PublishedTime, df.SourceName, df.SourceURL],
                    fill_color='lavender' ))
        ])

        return fig  
    else :
        df = pd.read_csv("../Data/NewsContent.csv", index_col=[0])
        df = df.loc[df["PublishedTime"] == news_date]
        column_list = df.columns.drop(["SourceURL"])
        fig = go.Figure(data=[go.Table(
            columnorder = [1,2,3,4,5],
            columnwidth = [80,150,130,80,80],
            header=dict(values=list(column_list),
                        fill_color='paleturquoise' ),
            cells=dict(values=[df.Title, df.Description, df.NewsURL, df.SourceName, df.PublishedTime],
                    fill_color='lavender' ))
        ])

        return fig

def unique_date():
    df = pd.read_csv("../Data/NewsContent.csv", index_col=[0])
    unique_date =  df.PublishedTime.unique().tolist()
    # unique_date.insert(0, "<all>")
    return unique_date

def sentiment_news_display(news_date): 

    df = pd.read_csv("../Data/NewsContentPreprocessed.csv", index_col=[0]).round(4)
    column_list = df.columns
    column_list = column_list.drop(["NewsURL", "SourceName", "SourceURL"])

    if news_date == "<all>":
        
        fig = go.Figure(data=[go.Table(
            columnorder = [1,2,3,4,5],
            columnwidth = [80,150,130,80,80],
            header=dict(values=list(column_list),
                        fill_color='white' ),
            cells=dict(values=[df.Title, df.Description, df.PublishedTime, 
                    df.SentimentValueTitle, df.SentimentValueDescription,
                    df.SentimentValueTitleVader , df.SentimentValueDescriptionVader
                    ],
                    fill_color='paleturquoise' ))
        ])

        return fig
    else : 
        df = df.loc[df["PublishedTime"] == news_date] 
        fig = go.Figure(data=[go.Table(
            columnorder = [1,2,3,4,5,6,7],
            columnwidth = [80,150,80,80,80,80,80],
            header=dict(values=list(column_list)),
            cells=dict(values=[df.Title, df.Description, df.PublishedTime, 
                    df.SentimentValueTitle, df.SentimentValueDescription,
                    df.SentimentValueTitleVader , df.SentimentValueDescriptionVader
                    ]))
        ])

        return fig

def sentiment_table():
    df = pd.read_csv("../Data/SentimentValues.csv").round(3)
    column_list=  ["Date","TextBlob Title",
    "TextBlob Desciption",
    "VADER Title",
    "VADER Description"]
    fig = go.Figure(data=[go.Table(
        columnorder = [1,2,3,4,5],
        columnwidth = [80,80,80,80,80],
        header=dict(values=list(column_list),
                    fill_color='yellowgreen',
                    line_color='darkslategray' ),
        cells=dict(values=[df.PublishedTime ,df.SentimentValueTitle, df.SentimentValueDescription,
                df.SentimentValueTitleVader , df.SentimentValueDescriptionVader
                ],
                line_color='darkslategray',
                fill_color=['coral','salmon','salmon','salmon','salmon' ]))
    ])

    return fig

def sentiment_graph():

        
    df = pd.read_csv("../Data/SentimentValues.csv")
    x_values  = list(df.PublishedTime.values)
    vaderTitle = list(df.SentimentValueTitleVader.values)
    TBDescription = list(df.SentimentValueDescription.values)
    y_values  = df.drop(["PublishedTime"],axis =1).agg("mean",axis =1)
    # fig = px.line(x = x_values, y = y_values, title='Life expectancy in Canada', width= 20)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, name='Average Sentiment',
                         line=dict(color='royalblue', width=4)))

    fig.add_trace(go.Scatter(x=x_values, y=vaderTitle, name='Vader Title Sentiment',
                         line=dict(color='salmon', width=1)))
    fig.add_trace(go.Scatter(x=x_values, y=TBDescription, name='TextBlob Description Sentiment',
                         line=dict(color='lightseagreen', width=1)))


    # Add figure title
    fig.update_layout(
        title_text="Average Sentiment "
    )

    # Set x-axis title
    fig.update_xaxes( title_text='Timestamp') 

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Senttiment Value</b>")

    return fig



############################## Section 2 ##############################
def sarima_prediction():
    overall_sarima = pd.read_csv("../Data/sarima_prediction_overall.csv", index_col = [0])
    x_value_overall = list(overall_sarima.index.values)
    y_value_overall_target = overall_sarima["InfectedCount"]
    y_value_overall_predicted = overall_sarima["Predicted SARIMA"]

    last7_sarima = pd.read_csv("../Data/sarima_prediction_last7.csv", index_col = [0])
    x_value_7 = list(last7_sarima.index.values)
    y_value_7_target = last7_sarima["InfectedCount"]
    y_value_7_predicted = last7_sarima["Predicted"]

    fig = make_subplots(rows=2, cols=1,subplot_titles=("Overview", "Last 7 Days"))

    fig.add_trace(
        go.Scatter(x=x_value_overall, y=y_value_overall_target,name = "Target Value"),
        
        row=1, col=1
    )


    fig.add_trace(
        go.Scatter(x=x_value_overall, y=y_value_overall_predicted, name = "SARIMA Prediction"),
        row=1, col=1
    )



    fig.add_trace(
        go.Scatter(x=x_value_7, y=y_value_7_target, name = "Target Value", line = dict(color = "blue")),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_value_7, y=y_value_7_predicted, name = "SARIMA Prediction", line = dict(color = "red")),
        row=2, col=1
    )


    fig.update_layout( autosize=True,height=600, title_text="SARIMA Prediction for last 7 days")

    
    fig.update_xaxes(automargin=True)
    return fig

    


def boosted_prediction():
    boosted_df = pd.read_csv("../Data/App/boosted_predict.csv")
    last_21_sentiment = pd.read_csv("../Data/App/sentiment_last_21.csv")

    sentiment_x_values = list(last_21_sentiment["Unnamed: 0"])
    last_21_sentiment = last_21_sentiment.drop(["Unnamed: 0", "SentimentValueDescription", "SentimentValueTitle"],axis = 1)
    sentiment_title = list(last_21_sentiment["SentimentValueTitleVader"])
    sentiment_description = list(last_21_sentiment["SentimentValueDescriptionVader"])

    boosted_df_x_values = list(boosted_df["Unnamed: 0"])
    boosted_df = boosted_df.drop(["Unnamed: 0"], axis = 1)
    boosted_df_y_values = list(boosted_df["0"])

    overall_sarima = pd.read_csv("../Data/sarima_prediction_overall.csv", index_col = [0])
    x_value_overall = list(overall_sarima.index.values)[-21:]
    y_value_overall_target = overall_sarima["InfectedCount"][-21:]
    y_value_overall_predicted = overall_sarima["Predicted SARIMA"][-21:]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=x_value_overall, y=y_value_overall_target,
                name = "Target Value", line = dict(width = 4)),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=x_value_overall, y=y_value_overall_predicted, 
                name = "SARIMA Prediction", line = dict(width = 4)),
        secondary_y=False
    )


    fig.add_trace(
        go.Scatter(x=boosted_df_x_values, y=boosted_df_y_values,
                name = "BoostedModel Prediction", line = dict(width = 4)),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x = sentiment_x_values, y=sentiment_title, 
                name="Title Sentiment",line = dict(width = 1, dash = 'dot', color = 'purple')),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x = sentiment_x_values, y=sentiment_description,
                name="Description Sentiment",line = dict(width = 1, dash = 'dot', color = 'grey')),
        secondary_y=True,
    )
    fig.update_layout(
        title_text="Overview for sentiment and predictions for the United States"
    )

    # Set x-axis title
    fig.update_xaxes(title_text='Timestamp') 

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Increased Count for the Infected</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Sentiment Value</b>", secondary_y=True)
    return fig


    
###################### Test Code ######################
###################### Deprecated######################

def demo():
    temp = [i for i in range(1, 100)]
    fig = px.line(temp, title='Life expectancy in Canada')
    # fig.show()
    return fig

def petal():
    df = px.data.iris() # iris is a pandas DataFrame
    fig = px.scatter(df, x="sepal_width", y="sepal_length")
    return fig