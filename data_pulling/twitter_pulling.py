import pandas as pd
import tweepy
import subprocess
import sys
from configurations.global_config import GlobalConfig

def fetch_tw(ids):
    list_of_tw_status = api.statuses_lookup(ids, tweet_mode= "extended")
    empty_data = pd.DataFrame()
    for status in list_of_tw_status:
            tweet_elem = {"tweet_id": status.id,
                     "screen_name": status.user.screen_name,
                     "tweet":status.full_text,
                     "date":status.created_at}
            empty_data = empty_data.append(tweet_elem, ignore_index = True)
    empty_data.to_csv(f"{company}_tweets.csv", mode="a")


if __name__ == '__main__':
    company = sys.argv[1]
    keyword = sys.argv[2]

    cmd = ''f'snscrape --max-results 100000 twitter-search "{keyword} since:2018-11-15 until:2020-11-15" > tw_{company}.txt'''
    process = subprocess.run(cmd, shell=True)

    access_token =GlobalConfig.ACCESS_TOKEN
    access_token_secret = GlobalConfig.ACCESS_TOKEN_SECRET
    consumer_key = GlobalConfig.CONSUMER_KEY
    consumer_secret = GlobalConfig.CONSUMER_SECRET

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    tweet_url = pd.read_csv(f".\\tw_{company}.txt", index_col= None, header = None, names = ["links"])

    af = lambda x: x["links"].split("/")[-1]
    tweet_url['id'] = tweet_url.apply(af, axis=1)

    ids = tweet_url['id'].tolist()

    total_count = len(ids)
    chunks = (total_count - 1) // 100 + 1

    for i in range(chunks):
        batch = ids[i*100:(i+1)*100]
        result = fetch_tw(batch)
