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
    # stock_puller = StockPuller(api_key=GlobalConfig.ALPHA_VANTAGE_API_KEY_EXTENDED_HISTORY,
    #                           ticker=GlobalConfig.NIKOLA_TICKER, interval=GlobalConfig.ONE_MIN_INTERVAL)
    # stock_puller.pull_data()

    # ==================== Parse Stock Data ====================
    # stock_parser = StockParser(ticker=GlobalConfig.TESLA_TICKER_STR, interval=GlobalConfig.ONE_MIN_INTERVAL)
    # rec_list = stock_parser.parse_stock_data()
    # stock_feature_dict = stock_parser.create_stock_feature_dict(single_stock_recording_list=rec_list,
    #                                                             start_time='09:00:00', end_time='17:00:00',
    #                                                             store_in_json_file=True)

    # ==================== Visualize Stock Data ====================
    # stock_visualizer = StockVisualizer(single_stock_recording_list=rec_list,
    #                                   stock_feature_dict=stock_feature_dict)
    # stock_visualizer.plot_all_in_one_chart()
    # stock_visualizer.plot_stock_features()
    # stock_visualizer.plot_stock_feature_histogranms()

    # ==================== Visualize Stock and Social Media Features ====================
    combined_visualizer = CombinedVisualizer(stock_ticker=GlobalConfig.PALANTIR_TICKER_STR)
    combined_visualizer.plot_sentiment_stock_feature_grafics(twitter_feature=GlobalConfig.TWITTER_NUM_POS,
                                                             reddit_feature=GlobalConfig.REDDIT_NUM_POS)
    combined_visualizer.plot_correlation_bar_chart(twitter_feature=GlobalConfig.TWITTER_NUM_POS,
                                                   reddit_feature=GlobalConfig.REDDIT_NUM_POS)
    combined_visualizer.plot_correlation_heatmap(use_time_lag=False,
                                                 time_lag_direction=GlobalConfig.TIME_LAG_RIGHT,
                                                 num_days=1)
    combined_visualizer.plot_distribution_comparison(perform_t_test=True,
                                                     stock_feature=GlobalConfig.MAX_MARGIN,
                                                     twitter_feature=GlobalConfig.TWITTER_NUM_POS,
                                                     reddit_feature=GlobalConfig.REDDIT_NUM_POS)


    # ending time
    ending_time = datetime.now()
    print(ending_time, ': Program finished.')

    # execution length
    print('Program took:', ending_time-starting_time, 'to run.')
