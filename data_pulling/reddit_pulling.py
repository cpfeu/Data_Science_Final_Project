from psaw import PushshiftAPI
import pandas as pd
import datetime as dt

api = PushshiftAPI()

target_subreddits = ['investing', 'wallstreetbets', 'wsb', 'stocks', 'SecurityAnalysis']
tickers = ['TSLA', 'KODK', 'HTZGQ', 'PLTR', 'SNOW', 'GS', 'BAC', 'NIO']

start_epoch = int(dt.datetime(2018, 1, 1).timestamp())

def fetch_reddit(query, subreddit, start_epoch):
    print(f'Fetching comment data for {query}...')
    gen = api.search_comments(q=query, subreddit=subreddit, after=start_epoch, filter=['body', 'created_utc', 'id', 'score', 'subreddit'])

    df = pd.DataFrame([thing.d_ for thing in gen])
    # df['Comment Date'] = dt.datetime.utcfromtimestamp(df['created_utc']).strftime('%Y-%m-%d %H:%M:%S')
    df.to_csv(query+'_reddit.csv')

for ticker in tickers:
    fetch_reddit(ticker, target_subreddits, start_epoch)