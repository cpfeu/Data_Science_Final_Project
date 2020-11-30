# Instructions for use:
# Call the script from the command line and pass two arguments
#    argument 1 = raw reddit comment file to preprocess, with comments in column named 'body' and comment timestamp in column named 'created_utc'
#    argument 2 = name of output directory 

import pandas as pd
import re, unicodedata
import contractions
from nltk import word_tokenize
from datetime import datetime
import sys
import os

df = pd.read_csv(sys.argv[1])

def to_datetime(ts):
    return datetime.utcfromtimestamp(int(ts)).strftime('%Y-%m-%d %H:%M:%S')

def to_lowercase(text):
    return text.lower()

def replace_contractions(text):
    return contractions.fix(text)

def remove_URL(text):
    return re.sub(r"http\S+", "", text)

def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def detokenize_words(words):
    separator = ' '
    return separator.join(words)

def preprocess_reddit(df, textcol='body', tscol='created_utc'):
    df[textcol] = df[textcol].astype(str)
    df[textcol] = df[textcol].apply(to_lowercase)
    df[textcol] = df[textcol].apply(replace_contractions)
    df[textcol] = df[textcol].apply(remove_URL)
    df[textcol] = df[textcol].apply(word_tokenize)
    df[textcol] = df[textcol].apply(remove_non_ascii)
    df[textcol] = df[textcol].apply(remove_punctuation)
    df[textcol] = df[textcol].apply(detokenize_words)
    df.dropna(inplace=True)
    df[tscol] = df[tscol].apply(to_datetime)
    return df

preprocess_reddit(df)

head, tail = os.path.split(sys.argv[1])

if not os.path.exists(sys.argv[2]):
    os.mkdir(sys.argv[2])

df.to_csv(sys.argv[2]+'/clean_'+tail, index=False)