#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import DataPreprocessingLibrary as dpl
import string
# https://pypi.org/project/COVID19Py/

pd.set_option('display.max_colwidth', -1)


API_COLUMNS = ["Title", "Description", "NewsURL", "PublishedTime", "SourceName" , "SourceURL"]
# api_content = pd.read_csv("../Data/NewsContentClean.csv")
api_content = pd.read_csv("../Data/NewsContent.csv", index_col=[0])


# In[2]:


from textblob import TextBlob
import nltk
from textblob import Word
from textblob.wordnet import VERB


# In[3]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# In[4]:


temp1 = api_content.iloc[-2].Title
print("Showing the results of the original : \n" + str(temp1) + "\n================================================\n")
print("Showing the results of the preprocessed : \n" + str(dpl.preprocess(temp1) + "\n================================================\n"))


# In[5]:


api_content["Title"] = api_content['Title'].apply(dpl.preprocess)
api_content["Description"] = api_content['Description'].apply(dpl.preprocess)

# api_content = api_content.drop(["Title","Description"] , axis = 1) 
# api_content.to_csv("../Data/NewsContentPreprocessed.csv", index= False)


# In[6]:


sentiment_value = []
sentiment_value_description = []

sentiment_value_nltk = []
sentiment_value_description_nltk = []

for title in api_content.Title.values:
    sentiment_value.append(  TextBlob( title ).sentiment.polarity  )
    sentiment_value_nltk.append(analyzer.polarity_scores(title)["compound"])
    
for description in api_content.Description.values:
    sentiment_value_description.append(  TextBlob( description ).sentiment.polarity  )
    sentiment_value_description_nltk.append(analyzer.polarity_scores(description)["compound"])
    


# In[7]:


api_content["SentimentValueTitle"] = sentiment_value
api_content["SentimentValueDescription"] = sentiment_value_description

api_content["SentimentValueTitleVader"] = sentiment_value_nltk
api_content["SentimentValueDescriptionVader"] = sentiment_value_description_nltk


# In[8]:


### General Sentiment value for the title is 0.03, pretty average but still more positive
### General Sentiment value for the description is 0.07, pretty average but still more positive


# In[9]:


api_content['SentimentValueTitle'].describe()


# In[10]:


api_content['SentimentValueDescription'].describe()


# In[11]:


api_content['SentimentValueTitleVader'].describe()


# In[12]:


api_content['SentimentValueDescriptionVader'].describe()


# In[13]:


sentiment_values_df = api_content.copy()
sentiment_values_df_grouped = sentiment_values_df.groupby("PublishedTime").mean()


# In[14]:


api_content.to_csv("../Data/NewsContentPreprocessed.csv")
sentiment_values_df_grouped.to_csv("../Data/SentimentValues.csv")


# In[15]:


pd.set_option('display.max_colwidth', 30)

api_content


# In[16]:


sentiment_values_df_grouped


# In[17]:


# temp  = pd.read_csv("../Data/SentimentValues.csv", index_col=[0])
# temp  = pd.read_csv("../Data/NewsContentPreprocessed.csv", index_col=[0])
# # temp


# In[18]:


showcase = pd.read_csv("../Data/NewsContent.csv", index_col=[0])
showcase["CleanTitle"] = showcase['Title'].apply(dpl.preprocess) 
showcase["CleanDescription"] = showcase['Description'].apply(dpl.preprocess) 
showcase = showcase.drop(["NewsURL" , "SourceName", "SourceURL"], axis = 1)
showcase


# In[19]:



dpl.make_word_cloud(showcase , "CleanTitle", "Title Wordcloud")
dpl.make_word_cloud(showcase, "CleanDescription", "News Content Wordcloud")

