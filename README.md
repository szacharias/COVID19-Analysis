# COVID-19 Analysis
This is my interpretation of the COVID19 data. <br>
The whole project will take up a span of 9 weeks. In this duration i will attempt to develop and optimize a deliverable software project. [Reference materials will be listed at the very bottom.](#Sources)

## Index

- [Description](#Description)

- [Goal](#Goal)

- [Objectives](#Objectives)

- [Environment](#Environment)

- [Findings Summary](#Summary)

- [Progress](#Progress)

- [Sources](#Sources)

## Goal


## Objectives
1. Implement time series model integrated with sentiment analysis to predict future outbreaks before hand, and even possibly when we can come out of qurantine. 

## Environment and Build
I am running on Ubuntu 19.10(x86_64) with Jupyter Lab(Jupyter notebook would suffice for testing the code) with python 3.7.4. Further inspections can be seen [here!](https://github.com/szacharias/COVID19-Analysis/blob/master/Code/Environment.ipynb) <br>
To start jupyter notebook/lab, input
```
jupyter notebook
jupyter lab
```
and navigate to [http://localhost:8888](http://localhost:8888). Port 8888 by default.
<br>
The required packages will be listed along the code in _requirements_ (provided later).
Run the following commands for the first time
```
pip install -r ./code/requirements.txt
```
 <br>


## Findings Summary 



## Progress
**Week 1, June 11th / June 18th**
- [X] Github Markdown
- [X] Read provided sources
- [X] Start implementations
**Week 2, June 19th / June 26th**
- [X] Data Cleaning
- [X] EDA
- [ ] Sentiment Analysis integrated with Time Series Analysis(how to quantify the sentiment value into the time series model)



## Sources 
- [Epidemic Modeling](https://medium.com/data-for-science/epidemic-modeling-101-or-why-your-covid19-exponential-fits-are-wrong-97aa50c55f8)

- [Epidemic Modeling 2 - some models are useful?](https://medium.com/data-for-science/epidemic-modeling-102-all-covid-19-models-are-wrong-but-some-are-useful-c81202cc6ee9)

- [Epidemic Modeling 3 - Confindence Intervals and Stochastic Effects](https://medium.com/data-for-science/epidemic-modeling-103-adding-confidence-intervals-and-stochastic-effects-to-your-covid-19-models-be618b995d6b)

- [Epidemic Modeling 4 - Seasonal Effects](https://medium.com/data-for-science/epidemic-modeling-104-impact-of-seasonal-effects-on-covid-19-16a1b14056f)

- [COVID-19 Data Source](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model)

- [Building your own COVID-19 Model](https://towardsdatascience.com/building-your-own-covid-19-epidemic-simple-model-using-python-e39788fbda55)

- [COVID-19 Visualization](https://towardsdatascience.com/visualise-covid-19-case-data-using-python-dash-and-plotly-e58feb34f70f)

- [Global Covid-19 Forecasting](https://towardsdatascience.com/global-covid-19-forecasting-with-linear-regression-and-arima-c154c163acc1)

- [Using Kalman Filter to Predict Coronavirus Spread](https://towardsdatascience.com/using-kalman-filter-to-predict-corona-virus-spread-72d91b74cc8)

- [Kalman Filters Sequel - Coronavirus Spread Prediction](https://medium.com/analytics-vidhya/coronavirus-updated-prediction-using-kalman-filter-3ef8b7a72409)

- [COVID-19 Predictions: A Machine Learning and Statistical Approach](https://medium.com/datadriveninvestor/covid-19-predictions-a-machine-learning-and-statistical-approach-410bef74f5c5)

- [COVID-19 and the first war of data science](https://medium.com/starschema-blog/covid-19-and-the-first-war-of-data-science-980798f075ef)

- [Time series infused with Event based algorithm](https://www.sciencedirect.com/science/article/pii/S0020025515000067)

- [Times series method in Python](https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/)

- [Deep Learning and Times Series](https://machinelearningmastery.com/start-here/#deep_learning_time_series)

- [Outbreak Prediction in Python](https://www.codespeedy.com/covid-19-outbreak-prediction-using-machine-learning-in-python/)

- [Using time series and sentiment analysis to forecast bitcoin prices](https://aisel.aisnet.org/cgi/viewcontent.cgi?article=1017&context=mcis2015)

- [Predicting Stock Returns using Sentiment analysis and LSTM](https://yujingma.com/2016/11/27/predicting-stock-returns-with-sentiment-analysis-and-lstm/)

- [Stock Prediction Using Twitter Sentiment Analysis](http://cs229.stanford.edu/proj2011/GoelMittal-StockMarketPredictionUsingTwitterSentimentAnalysis.pdf)
```
The technique used in this paper builds directly on the oneused by Bollen et al. [1]. 
The raw DJIA values are first fedinto the preprocessor to obtain the processed values. 
At thesame time, the tweets are fed to the sentiment analysis algo-rithm which outputs mood values for the four mood classesfor each day. 
These moods and the processed DJIA valuesare then fed to our model learning framework which usesSOFNN to learn a model to predict future DJIA values us-ing them. 
The learnt model as well as the previous DJIA andmood values are used by the portfolio management systemwhich runs the model to predict the future value and usesthe predicted values to make appropriate buy/sell decisions. 
```

- [Estimate Incubation Period of COVID19](https://www.acc.org/latest-in-cardiology/journal-scans/2020/05/11/15/18/the-incubation-period-of-coronavirus-disease)

- [How to Tune ARIMA Parameters in Python](https://machinelearningmastery.com/tune-arima-parameters-python/)

#### Datasets
- [John Hopkins Dataset](https://github.com/CSSEGISandData/COVID-19/)

- [Dataset](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/)

- [Model Code](https://github.com/DataForScience/Epidemiology101/blob/master/Epidemiology001.ipynb)

- [Berkeley COVID-19 Dashboard](https://covidvis.berkeley.edu/?fbclid=IwAR3Tax0-IzkqEZ_7ZPyddaMhIoEDnSDTsetxLCK57arcxALggCrxgi6zRmY)

- [New York Times](https://github.com/nytimes/covid-19-data)

- [Enhanced Dataset](https://github.com/covidvis/covid19-vis/blob/master/data/quarantine-activity-US-Apr16-long.csv)

- [Cleaned data for COVID19 analysis - starschema](https://github.com/starschema/COVID-19-data)

- [Snowflake - dataset updated daily](https://www.snowflake.com/datasets/starschema/) 

- [COVID-19 Chest XRay Images](https://github.com/ieee8023/covid-chestxray-dataset)

- [Kaggle Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
#### Libraries

- [qwikidata](https://qwikidata.readthedocs.io/en/stable/readme.html)

- [Outlier Detection](https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/)