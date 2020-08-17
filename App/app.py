
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
from datetime import datetime, timedelta




############ Initial Loading ############
baseLocation = "../Data/"

datetime_format = "%Y-%m-%d"

today = datetime.now().strftime(datetime_format)

# baseURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
# image_directory =  os.getcwd()
# image_filename = image_directory + '/FinalModelPrediction.png' 
# encoded_image = base64.b64encode(open(image_filename, 'rb').read())


## Read from URL // Deprecated
# cumulated_confirmed     = pd.read_csv(cumulated_confirmed_url)
# cumulated_deaths        = pd.read_csv(cumulated_deaths_url)
# cumulated_recovered     = pd.read_csv(cumulated_recovered_url)
# cumulated_confirmed_url = baseURL + "time_series_covid19_confirmed_global.csv"
# cumulated_deaths_url    = baseURL + "time_series_covid19_deaths_global.csv"
# cumulated_recovered_url = baseURL + "time_series_covid19_recovered_global.csv"


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


############ Initial Loading End ############


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

tickFont = {'size':12, 'color':"rgb(30,30,30)", 'family':"Semi-bold 600"}

markdown_introduction = '''
#### Matthew Sah
 <msah@buffalo.edu> &nbsp; [Github](https://github.com/szacharias/COVID19-Analysis) &nbsp; [Linkedin](https://www.linkedin.com/in/matthew-sah/) 
 ---
'''

markdown_index = '''
### Index
1. Raw Time Series Data
2. News Data & Sentiment data
3. Prediction Graph
---
'''

markdown_chapter_1 = '''
## Section 1 : Raw Time Series Data
'''
markdown_chapter_2 = '''
## Section 2 : News Data & Sentiment data
'''
markdown_chapter_3 = '''
## Section 3 : Prediction Graph
'''
markdown_split = '''
---
'''

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    style={ 'font-family':"Courier New, monospace" },
    children=[
        html.H1('Coronavirus Infection Prediction in the United States'),
        dcc.Markdown(children = markdown_introduction),
        dcc.Markdown(children = markdown_index),
        ################# Chapter 1 #################
        dcc.Markdown(children = markdown_chapter_1),
        html.Div(className="row", children=[
            html.Div(className="three columns", children=[
                html.H5('Country'),
                dcc.Dropdown(
                    id='country',
                    options=[{'label':c, 'value':c} for c in unique_countries],
                    value='US'
                )
            ]),
            html.Div(className="three columns", children=[
                html.H5('State / Province'),
                dcc.Dropdown(
                    id='state'
                )
            ]) 
        ]),
        html.Div(className = "row", children = [
            html.Div(className = "six columns", children = [
                html.H5("SI Graph"),
                dcc.Graph(
                    id="SI_graph"
                )
            ]),
            html.Div(className = "six columns", children = [
                html.H5("Increase In Infected : 7 Day moving Difference"),
                dcc.Graph(
                    id="infected_graph"
                )
            ])

        ]),

        ################# Chapter 2 #################
        dcc.Markdown(children = markdown_split),
        dcc.Markdown(children = markdown_chapter_2),
        html.H4("1. News by Date"),
        #### News' Date select
        html.Div(className="row", children=[
            html.Div(className="three columns", children=[
                html.H5('Date'),
                dcc.Dropdown(
                    id='news_date',
                    options=[{'label':c, 'value':c} for c in gf.unique_date()]
                    ,

                    value=gf.unique_date()[-1]

                )
            ])
        ]),
        dcc.Graph(id = "news_display"),


        #### News' Date select for sentiment
        html.H4( "2. Processed News and it's Sentiment"),
        html.Div(className="row", children=[
            html.Div(className="three columns", children=[
                html.H5('Date'),
                dcc.Dropdown(
                    id='news_sentiment_date',
                    options=[{'label':c, 'value':c} for c in gf.unique_date()]
                    ,

                    value=gf.unique_date()[-1]

                )
            ])
        ]),
        dcc.Graph(id = "news_sentiment_display"),


        #### Group by date sentiment
        html.H4( "3. Average Sentiment by Day and it's trend"),
        
        html.Div(className = "row", children = [
            html.Div(className = "five columns", children = [
                html.H5("Sentiment Grouped by day"),
                dcc.Graph(
                    id="sentiment_table",
                    figure = gf.sentiment_table()
                )
            ]),
            html.Div(className = "seven columns", children = [
                html.H5("Sentiment Trends"),
                dcc.Graph(
                    id="sentiment_graphs",
                    figure = gf.sentiment_graph()
                )
            ])

        ]),
        
        
        ################# Chapter 3 #################
        dcc.Markdown(children = markdown_split),
        dcc.Markdown(children = markdown_chapter_3),
        html.H4( "1. SARIMA Initial Prediction"),
        dcc.Graph(
                    id="SARIMA_Prediction",
                    figure = gf.sarima_prediction()
                ),

        html.H4( "2. Final Algorithm"),
        dcc.Graph(
                    id="boosted_prediction",
                    figure = gf.boosted_prediction()
                )

        #, dcc.Graph(id = "demo", figure = gf.demo())

    ]
)
############################## Section 1 ##############################
############### Update State Options ###############
@app.callback(
    [Output('state', 'options'), Output('state', 'value')],
    [Input(component_id='country', component_property='value')]
)
def update_states(country):
    return gf.update_states(country)

############### Update Base Infected Graph ###############

@app.callback(
    Output('SI_graph', 'figure'), 
    [Input('country', 'value'), Input('state', 'value')]
)
def update_graph(country, state): 
    return gf.update_graph(country, state)

############### Update Increase Infected Graph ###############

@app.callback(
    Output('infected_graph', 'figure'), 
    [Input('country', 'value'), Input('state', 'value')]
)
def update_increase_graph(country, state): 
    return gf.update_increase_graph(country, state)
    # return gf.update_increased_graph(country, state)

############################## Section 2 ##############################
############### Update News Date Options ###############
@app.callback(
    Output('news_display', 'figure'),
    [Input(component_id='news_date', component_property='value')]
)
def news_display(news_date):
    return gf.news_display(news_date)

@app.callback(
    Output('news_sentiment_display', 'figure'),
    [Input(component_id='news_sentiment_date', component_property='value')]
)
def sentiment_news_display(news_sentiment_date):
    return gf.sentiment_news_display(news_sentiment_date)


############################## Section 3 ##############################


############### Server Settings ###############


server = app.server

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug = True)

