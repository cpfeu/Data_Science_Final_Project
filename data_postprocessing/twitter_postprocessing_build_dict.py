import sys
import os
import pandas as pd
from datetime import datetime
import json

df = pd.read_csv(sys.argv[1])

df.drop(df.columns[[1,3]], axis=1, inplace=True)

def to_datetime(ts):
    return datetime.strptime(ts, '%m/%d/%Y %H:%M')

def to_date(ts):
    return datetime.date(ts)

def to_time(ts):
    return datetime.time(ts)

df['datetime'] = df['date'].apply(to_datetime)
df['date'] = df['datetime'].apply(to_date)
df['time'] = df['datetime'].apply(to_time)
df.drop('datetime', axis=1, inplace=True)
df['date'] = df['date'].astype(str)
df['time'] = df['time'].astype(str)

date_lst = df['date'].unique()

d = dict()

for i in date_lst:
    d.update({str(i):dict()})
    temp_df = df[df['date'].astype(str)==str(i)]
    tweets = temp_df['body'].tolist()
    predictions = temp_df['prediction'].tolist()
    sentiments = temp_df['sentiment_score'].tolist()
    timestamps = temp_df['time'].tolist()
    d.get(i).update({'tweets':tweets, 'predictions':predictions, 'sentiments':sentiments, 'timestamps':timestamps})

head, tail = os.path.split(sys.argv[1])
filename, file_extension = os.path.splitext(tail)

if not os.path.exists(sys.argv[2]):
    os.mkdir(sys.argv[2])

with open(sys.argv[2]+'/'+filename+'_dict.json', 'w') as fp:
    json.dump(d, fp)