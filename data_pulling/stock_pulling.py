import os
import csv
import requests
from datetime import datetime
from configurations.global_config import GlobalConfig


class StockPuller:

    def __init__(self, api_key=GlobalConfig.ALPHA_VANTAGE_API_KEY_EXTENDED_HISTORY,
                 ticker=GlobalConfig.TESLA_TICKER_STR,
                 interval=GlobalConfig.ONE_MIN_INTERVAL):
        self.api_key = api_key
        self.ticker = ticker
        self.interval = interval



    def pull_data(self):

        print(datetime.now(), ': <pull_data> function for', self.ticker, 'started...')

        # pull stock data set (2 years of trading)
        data_batch_list = []
        for slice in GlobalConfig.SLICE_LIST:
            data = requests.get('https://www.alphavantage.co/query?'
                                'function=TIME_SERIES_INTRADAY_EXTENDED&'
                                'symbol={}&interval={}&'
                                'slice={}&apikey={}'.format(self.ticker, self.interval, slice, self.api_key))
            data_content = data.content
            data_content_str = data_content.decode('utf-8')
            data_batch_list.append(data_content_str)
            print('Pulled data for', self.ticker, 'at', slice)

        # store in csv file
        os.makedirs(os.path.join(GlobalConfig.DATA_BASE_PATH, self.ticker), exist_ok=True)
        with open(os.path.join(GlobalConfig.DATA_BASE_PATH, self.ticker,
                               self.ticker+'_'+self.interval+'.csv'), 'w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=',')
            for idx, data_batch in enumerate(data_batch_list):
                if idx == 0:
                    for line in data_batch.split('\r\n'):
                        csv_writer.writerow(line.split(','))
                else:
                    for idx, line in enumerate(data_batch.split('\r\n')):
                        if idx == 0:
                            continue
                        else:
                            csv_writer.writerow(line.split(','))

        print(datetime.now(), ':', self.ticker, 'stock data successfully pulled and stored.')






