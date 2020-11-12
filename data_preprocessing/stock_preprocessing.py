import os
import math
import pandas as pd
from datetime import datetime
from configurations.global_config import GlobalConfig


class StockParser:
    def __init__(self, ticker=GlobalConfig.TESLA_TICKER_STR, interval=GlobalConfig.ONE_MIN_INTERVAL):
        self.ticker = ticker
        self.interval = interval
        os.makedirs(os.path.join(GlobalConfig.PROGRAMMING_OUTPUTS_BASE_PATH, ticker), exist_ok=True)

    class SingleStockRecording:
        def __init__(self, name, time_stamp, open, high, low, volume, close):
            self.name = name
            self.time_stamp = time_stamp
            self.open = open
            self.high = high
            self.low = low
            self.volume = volume
            self.close = close


    def parse_stock_data(self):
        data_pd = pd.read_csv(filepath_or_buffer=os.path.join(GlobalConfig.DATA_BASE_PATH,
                                                              self.ticker,
                                                              self.ticker+'_'+self.interval+'.csv'),
                               sep=',', header=0, index_col=False)
        data_dict = data_pd.to_dict(orient='list')
        single_stock_recording_list = []
        for ts_idx in range(0, len(list(data_dict.get(GlobalConfig.STOCK_PARAM_TIME)))):
            if not math.isnan(data_dict.get(GlobalConfig.STOCK_PARAM_OPEN)[ts_idx]):
                time_stamp = data_dict.get(GlobalConfig.STOCK_PARAM_TIME)[ts_idx]
                open_val = data_dict.get(GlobalConfig.STOCK_PARAM_OPEN)[ts_idx]
                close_val = data_dict.get(GlobalConfig.STOCK_PARAM_CLOSE)[ts_idx]
                high_val = data_dict.get(GlobalConfig.STOCK_PARAM_HIGH)[ts_idx]
                low_val = data_dict.get(GlobalConfig.STOCK_PARAM_LOW)[ts_idx]
                volume_val = data_dict.get(GlobalConfig.STOCK_PARAM_VOLUME)[ts_idx]
                single_stock_recording_list.append(self.SingleStockRecording(name=self.ticker,
                                                                             time_stamp=datetime.strptime(time_stamp,
                                                                                                          '%Y-%m-%d %H:%M:%S'),
                                                                             open=open_val,
                                                                             high=high_val,
                                                                             low=low_val,
                                                                             volume=volume_val,
                                                                             close=close_val))
        print(datetime.now(), ':', self.ticker, 'stock parsing completed.')

        return single_stock_recording_list









