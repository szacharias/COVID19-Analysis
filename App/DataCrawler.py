#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json
import time
import pandas as pd
import numpy as np
from WebCrawlerLibrary import gen_query_url, parse_response, extract_data_from_api, query_start_date
from datetime import datetime, timedelta


# In[2]:


# Design dataframe
API_COLUMNS = ["Title", "Description", "NewsURL", "PublishedTime", "SourceName" , "SourceURL"]

api_content = pd.read_csv("../Data/NewsContent.csv", index_col=[0])
# api_content = pd.read_csv("../Data/NewsContent.csv")


# In[3]:


api_content_current_session = pd.DataFrame(columns=API_COLUMNS)


# In[5]:


# query_index_date.strftime(datetime_format) != datetime.now().strftime(datetime_format)


# In[6]:


# Process query days pipeline
datetime_format = "%Y-%m-%d"
# start_date = datetime(2020,1,29)
start_date = query_start_date(api_content)
# end_date =  datetime(2020,2,1)

query_index_date = start_date 

status_code = 200
counter = 0
max_count = 49
first_time = False


while counter < max_count and status_code == 200 and query_index_date.strftime(datetime_format) != datetime.now().strftime(datetime_format):
    # counter does not surpass 100 queryies per day
    # status_code is still successful
    # index date has yet to surpass today
    
    # Generate URLs
    query_index_date_next = query_index_date + timedelta(days=1)
    url = gen_query_url(query_index_date.strftime(datetime_format), query_index_date_next.strftime(datetime_format) )
    
    # Parse URLs
    response = requests.get(url)
    
    # get Status_code and content
    status_code, output_content = parse_response(response)
    
    # Parse content
    if status_code == 200 and output_content != "no returns":    
        new_api_data = extract_data_from_api(output_content)
    
    if counter == 0 and first_time == True:
        api_content = pd.DataFrame(new_api_data, columns = API_COLUMNS)
#         print(api_content)
    else :
        new_api_df = pd.DataFrame(new_api_data, columns = API_COLUMNS)
#         print(new_api_df)
        api_content = api_content.append(new_api_df, ignore_index = True).drop_duplicates()
        api_content_current_session = api_content_current_session.append(new_api_df, ignore_index = True).drop_duplicates()
    

    # Post Processing conditions
    query_index_date = query_index_date_next
    counter += 1
    if counter == max_count:
        print("Has Reached Max Count of " + str(counter))
    if status_code != 200:
        print("Status Code " + str(status_code))
    time.sleep(3)


# In[8]:


# temp = api_content_current_session.copy()
# temp = temp.drop_duplicates(["Title" , "PublishedTime"])
# api_content = api_content.append(temp, ignore_index = True).drop_duplicates()


# In[7]:


api_content.PublishedTime   = pd.to_datetime(api_content.PublishedTime, utc=True).dt.strftime('%Y-%m-%d')
api_content_current_session.PublishedTime   = pd.to_datetime(api_content_current_session.PublishedTime, utc=True).dt.strftime('%Y-%m-%d')

api_content = api_content.drop_duplicates(["Title", "PublishedTime"]).sort_values(["PublishedTime"])
api_content


# In[10]:


api_content_current_session.to_csv("../Data/NewContentCurrentSession.csv")
api_content.to_csv("../Data/NewsContent.csv")


# ---

# In[9]:



# import GetOldTweets3 as gt


# https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset

# In[102]:


# temp1 = temp_cur_session.drop_duplicates(["Title","PublishedTime"])
# api_content_current_session = temp1
# temp_cur_session

