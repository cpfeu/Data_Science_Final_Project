import os
import math
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
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

        return single_stock_recording_list[::-1]





    def extract_value_lists_from_ssr_list(self, single_stock_recording_list):

        open_list = []
        close_list = []
        high_list = []
        low_list = []
        volume_list = []
        for ssr in single_stock_recording_list:
            if not math.isnan(ssr.open):
                open_list.append(ssr.open)
                close_list.append(ssr.close)
                high_list.append(ssr.high)
                low_list.append(ssr.low)
                volume_list.append(ssr.volume)

        return open_list, close_list, high_list, low_list, volume_list





    def prepare_for_json(self, dict_item):

        prepared_dict = dict()
        for date_key, date_dict in dict_item.items():

            # convert datetime.date key into str key
            prepared_dict.update({str(date_key): dict()})

            for feature_key, feature_value in date_dict.items():

                # convert datetime.time value into str value
                if feature_key == GlobalConfig.START_TIME or feature_key == GlobalConfig.END_TIME:
                    prepared_dict.get(str(date_key)).update({feature_key: str(feature_value)})
                else:
                    prepared_dict.get(str(date_key)).update({feature_key: feature_value})

        return prepared_dict





    def create_stock_feature_dict(self, single_stock_recording_list, start_time, end_time,
                                  store_in_json_file=False):

        # create list with all relevant days in datetime format
        start_date = single_stock_recording_list[0].time_stamp.date()
        end_date = single_stock_recording_list[-1].time_stamp.date()
        date_list = []
        for day_counter in range(int((end_date - start_date).days)+1):
            date_list.append(start_date + timedelta(days=day_counter))


        # distribute single_stock_recordings to dictionary
        single_stock_recording_date_dict = dict()
        for date in date_list:
            single_stock_recording_date_dict.update({date: []})
        for single_stock_recording in single_stock_recording_list:
            single_stock_recording_date_dict.get(single_stock_recording.time_stamp.date()).append(single_stock_recording)


        # strip start_time and end_time to datetime format
        start_time_dt = datetime.strptime(start_time, "%H:%M:%S").time()
        end_time_dt = datetime.strptime(end_time, "%H:%M:%S").time()


        # create stock feature dictionary
        stock_feature_dict = dict()
        for date in date_list[1:]:
            stock_feature_dict.update({date: dict()})

            # add start_time and end_time to feature dict
            stock_feature_dict.get(date).update({GlobalConfig.START_TIME: start_time_dt})
            stock_feature_dict.get(date).update({GlobalConfig.END_TIME: end_time_dt})

            # add single_stock_recording_list to feature dict
            ssr_list_feature = []
            ssr_list_date = single_stock_recording_date_dict.get(date)
            for ssr in ssr_list_date:
                if ssr.time_stamp.time() >= start_time_dt and ssr.time_stamp.time() <= end_time_dt:
                    ssr_list_feature.append(ssr)
                if not store_in_json_file:
                    stock_feature_dict.get(date).update({GlobalConfig.SSR_LIST: ssr_list_feature})

            # ========== create temporary open, close, high, low, volume lists ==========
            open_list, close_list, high_list, low_list, volume_list = \
                self.extract_value_lists_from_ssr_list(ssr_list_feature)

            # other features depend on whether there is data present for the current date
            if len(open_list) > 0:

                # add over over night difference to feature dict
                try:
                    over_night_diff = single_stock_recording_date_dict.get(date)[0].open - \
                                      single_stock_recording_date_dict.get(date - timedelta(days=1))[-1].close
                    stock_feature_dict.get(date).update({GlobalConfig.OVER_NIGHT_DIFF: over_night_diff})
                except IndexError:
                    stock_feature_dict.get(date).update({GlobalConfig.OVER_NIGHT_DIFF: 0})

                # add max_margin feature to feature dict
                max_margin = np.amax(high_list) - np.amin(low_list)
                stock_feature_dict.get(date).update({GlobalConfig.MAX_MARGIN: max_margin})

                # add absolute difference to feature dict
                abs_difference = close_list[-1] - open_list[0]
                stock_feature_dict.get(date).update({GlobalConfig.ABS_DIFFERENCE: abs_difference})

                # add percentage change to feature dict
                per_change = ((close_list[-1] - open_list[0]) / open_list[0]) * 100
                stock_feature_dict.get(date).update({GlobalConfig.PER_CHANGE: per_change})

                # add absolute trading volume to feature dict
                abs_vol = np.sum(volume_list)
                stock_feature_dict.get(date).update({GlobalConfig.ABS_VOL: abs_vol})

                # add volume fluctuation to feature dict
                vol_fluc = np.std(volume_list)
                stock_feature_dict.get(date).update({GlobalConfig.VOL_FLUC: vol_fluc})

                # add price fluctuation to feature dict
                price_fluc = np.std(high_list)
                stock_feature_dict.get(date).update({GlobalConfig.PRICE_FLUC: price_fluc})

            else:

                #add over night difference to feature dict
                stock_feature_dict.get(date).update({GlobalConfig.OVER_NIGHT_DIFF: 0})

                # add max_margin feature to feature dict
                stock_feature_dict.get(date).update({GlobalConfig.MAX_MARGIN: 0})

                # add absolute difference to feature dict
                stock_feature_dict.get(date).update({GlobalConfig.ABS_DIFFERENCE: 0})

                # add percentage change to feature dict
                stock_feature_dict.get(date).update({GlobalConfig.PER_CHANGE: 0})

                # add absolute trading volume to feature dict
                stock_feature_dict.get(date).update({GlobalConfig.ABS_VOL: 0})

                # add volume fluctuation to feature dict
                stock_feature_dict.get(date).update({GlobalConfig.VOL_FLUC: 0})

                # add price fluctuation to feature dict
                stock_feature_dict.get(date).update({GlobalConfig.PRICE_FLUC: 0})

        print(datetime.now(), ':', self.ticker, 'stock_feature_dict created.')

        if store_in_json_file:
            stock_feature_dict_json = self.prepare_for_json(dict_item=stock_feature_dict)
            with open('../feature_dictionaries/'+self.ticker+'_stock_feature_dict.json',
                      'w') as file:
                json.dump(stock_feature_dict_json, file)

        return stock_feature_dict







