import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt



def preprocess(in_text):
    # If we have html tags, remove them by this way:
    #out_text = remove_tags(in_text)
    # Remove punctuations and numbers
    out_text = re.sub('[^a-zA-Z]', ' ', in_text)
    # Convert upper case to lower case
    out_text="".join(list(map(lambda x:x.lower(),out_text)))
    # Remove single character
    out_text = re.sub(r"\s+[a-zA-Z]\s+", ' ', out_text)
    
    out_text = re.sub("coronavirus", "virus", out_text)
    
    out_text = re.sub("corona", "virus", out_text)
    
    out_text = re.sub("covid19", "virus", out_text)
    
    out_text = re.sub("covid", "virus", out_text)
    
    out_text = re.sub("trump", "president", out_text)
    
    return out_text



def remove_punctuation(text):
    return re.sub(r'[^\w\s]','',text)

def remove_single_character(text):
    return re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

####### WordCloud ######
def make_word_cloud(showcase , query_column, title):
    
    ##################### Counting Words ################
    from collections import Counter
    results = Counter()
    temp = pd.DataFrame(showcase[query_column])
    showcase[query_column].str.split().apply(results.update)
    results.most_common(100)[:5]
    
    
    ##################### Stop Words Removal ################
    from wordcloud import STOPWORDS
#     pop_list = ["in" ,"to", "of" , "as", "for", "on", "and",
#                 "is", "the", "from", "at" , "with",
#                "it" , "by" , "that", "has" , "are", "have",
#                 "will"
#            ]
    stopwords = set(STOPWORDS)
    pop_list= ["will"]
    for words in stopwords:
        pop_list.append(words)
    pop_list = set(pop_list)

    ##################### Remove stopwords from most common words ################
    ##################### and refresh value based on index ################
    temp = results.most_common(100)
    temp_strings =  {item[0] : len(temp) - index  for index, item in enumerate(temp)}
    for items in pop_list: 
        try:
            temp_strings.pop(items)
        except Exception as err :
            continue
        
    ##################### Word Cloud ################
    from wordcloud import WordCloud 

    # Stopwords dont work for generate from frequencies
#     stopwords = set(STOPWORDS)b
#     stopwords.update(["the", "us", "to", "in", "virus", "president"])
    
    wordcloud = WordCloud(width=1800,height=1000, max_words=200,stopwords=stopwords,
                          background_color="white",relative_scaling=1,
                          normalize_plurals=True).generate_from_frequencies(temp_strings)
    
    plt.figure(figsize = (10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.savefig("./img/WordCloud_" + str(title) + ".png")
    # plt.show()   