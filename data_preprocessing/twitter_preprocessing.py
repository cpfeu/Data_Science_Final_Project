import pandas as pd
import numpy as np


def to_datetime(s):
    from datetime import datetime
    s = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    
    return s

def clean(df):
    
    from nltk.corpus import stopwords
    import re
    from googletrans import Translator
    from language_detector import detect_language
    
    #remove $ and @mentions
    df['tweet'] = df['tweet'].map(lambda x: re.sub(r'\$[A-Za-z0-9]*','',x))
    #df['tweet'] = df['tweet'].map(lambda x: re.sub(r'\#[A-Za-z0-9]*','',x))
    df['tweet'] = df['tweet'].map(lambda x: re.sub(r'\@[A-Za-z0-9]*','',x))   
    
    #to datetime format
    df.drop( df[df['date'] == 'date'].index , inplace=True)
    df['date'] = df['date'].apply(lambda x: to_datetime(x) if type(x)== str else x)
    
    #drop first column 
    df= df.drop(df.columns[0], axis=1)
    
    #remove https: links
    df['tweet'] = df['tweet'].str.replace(r'https?://[^\s<>"]+|www\.[^\s<>"]+', "")
    #df['tweet'] = df['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    
    #remove punctuation 
    df['tweet'] = df['tweet'].str.replace('[^\w\s]','')
    
    #to lowercase
    df['tweet'] = df['tweet'].map(lambda s:s.lower() if type(s) == str else s)
    
    #delete line breaks
    df['tweet'].replace(r'\s+|\\n', ' ', regex=True, inplace=True) 
    
    df= df.reset_index(drop=True)
    
    #drop all foreign languages
    '''for x in range(len(df['tweet'])):
        if detect_language(df['tweet'][x]) != 'English':
            df = df.drop(index= x)
        else: 
            continue'''
    
    df['tweet'] = df['tweet'].map(lambda x: x.replace(x,'') if detect_language(x) != 'English' else x)
    df['tweet'].replace('', np.nan, inplace=True)
    
    #drop nulls
    df= df.dropna()
    
    df= df.reset_index(drop=True)
    
    ''' #remove stop words
    stop = stopwords.words('english')
    df['tweet'] = df['tweet'].apply(lambda x: [w.strip() for w in x if w.strip() not in stop])
    df['tweet'].apply(lambda x: [item for item in x if item not in stop])'''
    
    return df
