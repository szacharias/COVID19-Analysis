import pandas as pd
import numpy as np
import re

def preprocess(in_text):
    # If we have html tags, remove them by this way:
    #out_text = remove_tags(in_text)
    # Remove punctuations and numbers
    out_text = re.sub('[^a-zA-Z]', ' ', in_text)
    # Convert upper case to lower case
    out_text="".join(list(map(lambda x:x.lower(),out_text)))
    # Remove single character
    out_text = re.sub(r"\s+[a-zA-Z]\s+", ' ', out_text)
    return out_text


def remove_punctuation(text):
    return re.sub(r'[^\w\s]','',text)

def remove_single_character(text):
    return re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
