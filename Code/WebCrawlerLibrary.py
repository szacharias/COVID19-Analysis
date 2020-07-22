import requests
import json
import time
import pandas as pd
import numpy as np 
from datetime import datetime, timedelta


def extract_data_from_api(output):
    return_list = []
    all_news = output["articles"]
    for single_news in all_news: 
        
#         print("Title :"         + single_news["title"])
#         print("Description :"   + single_news["description"])
#         print("URL :"           + single_news["url"])
#         print("PublishedTime :" + single_news["publishedAt"])
#         print("SourceName : "   + single_news["source"]["name"])
#         print("SourceURL : "    + single_news["source"]["url"]) 

        return_list_single = []
        return_list_single.append(single_news["title"])
        return_list_single.append(single_news["description"])
        return_list_single.append(single_news["url"])
        return_list_single.append(single_news["publishedAt"])
        return_list_single.append(single_news["source"]["name"])
        return_list_single.append(single_news["source"]["url"]) 
        return_list.append(return_list_single)    
        
    return return_list

# api_data = extract_data_from_api(output)


def parse_response(response):
    
    output = "no returns"
    if response.status_code == 200:

    #   Binary response
        binary = response.content

    #   Unformatted output
        output = json.loads(binary)

    #   Formatted output
#         print(json.dumps(output, sort_keys = True, indent = 4))

    else :
        print("Query Error : " + str(response.status_code))
    
    return response.status_code, output

def gen_query_url(mindate, maxdate):
    ### https://gnews.io/docs/v3?shell#http-request
    url_base = "https://gnews.io/api/v3/search?"

    query = {
        "token":"837a1cfd8cff3793dca10c2b9478dcce",
        "lang":"en",
        "country":"us",
        "mindate": mindate,
        "maxdate": maxdate,
        "in":"title"


    }
    from urllib import parse
    query_value = parse.urlencode(query)

    url = url_base +  "q=covid19|coronavirus us"+"&"+ query_value 
    print(url)
    return url

# response = requests.get(url)

# ### https://gnews.io/docs/v3?shell#http-request
# url_base = "https://gnews.io/api/v3/search?"

# query = {
#     "token":"837a1cfd8cff3793dca10c2b9478dcce",
#     "lang":"en",
#     "country":"us",
#     "mindate":"2020-03-10",
#     "maxdate":"2020-03-17",
#     "in":"title"
    
    
# }
# from urllib import parse
# query_value = parse.urlencode(query)

# url = url_base +  "q=covid19|coronavirus us"+"&"+ query_value 
# response = requests.get(url)