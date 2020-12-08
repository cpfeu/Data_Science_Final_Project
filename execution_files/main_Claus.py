from datetime import datetime
from data_pulling.stock_pulling import StockPuller
from configurations.global_config import GlobalConfig
from data_preprocessing.stock_preprocessing import StockParser
from visualizations.stock_visualization import StockVisualizer
from visualizations.social_media_stock_visualization import CombinedVisualizer




if __name__ == '__main__':

    # starting time
    starting_time = datetime.now()
    print(starting_time, ': Program started.')


    # ==================== Pull Stock Data ====================
    #stock_puller = StockPuller(api_key=GlobalConfig.ALPHA_VANTAGE_API_KEY_EXTENDED_HISTORY,
    #                           ticker=GlobalConfig.HERTZ_TICKER_STR, interval=GlobalConfig.ONE_MIN_INTERVAL)
    #stock_puller.pull_data()

    # ==================== Parse Stock Data ====================
    # stock_parser = StockParser(ticker=GlobalConfig.BANK_OF_AMERICA_TICKER, interval=GlobalConfig.ONE_MIN_INTERVAL)
    # rec_list = stock_parser.parse_stock_data()
    # stock_feature_dict = stock_parser.create_stock_feature_dict(single_stock_recording_list=rec_list,
    #                                                             start_time='09:00:00', end_time='17:00:00',
    #                                                             store_in_json_file=False)

    # for i in date_lst:
    #     HTZ.update({str(i): dict()})
    #     temp_df = df[df['date'].astype(str) == str(i)]
    #     tweets = temp_df['body'].tolist()
    #     predictions = temp_df['prediction'].tolist()
    #     sentiments = temp_df['sentiment_score'].tolist()
    #     timestamps = temp_df['time'].tolist()
    #     HTZ.get(i).update({'tweets': tweets, 'predictions': predictions, 'sentiments': sentiments})

    # ==================== Visualize Stock Data ====================
    # stock_visualizer = StockVisualizer(single_stock_recording_list=rec_list,
    #                                   stock_feature_dict=stock_feature_dict)
    # stock_visualizer.plot_all_in_one_chart()
    # stock_visualizer.plot_stock_features()
    # stock_visualizer.plot_stock_feature_histogranms()

    # ==================== Visualize Stock and Social Media Features ====================
    combined_visualizer = CombinedVisualizer(stock_ticker=GlobalConfig.TESLA_TICKER_STR)
    # combined_visualizer.plot_sentiment_stock_feature_grafics()
    # combined_visualizer.plot_correlation_bar_chart()
    combined_visualizer.plot_correlation_heatmap(rescale=False)

    # ending time
    ending_time = datetime.now()
    print(ending_time, ': Program finished.')

    # execution length
    print('Program took:', ending_time-starting_time, 'to run.')
