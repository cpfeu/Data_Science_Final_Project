from datetime import datetime
from data_pulling.stock_pulling import StockPuller
from configurations.global_config import GlobalConfig
from data_preprocessing.stock_preprocessing import StockParser
from visualizations.stock_visualization import StockVisualizer




if __name__ == '__main__':

    # starting time
    starting_time = datetime.now()
    print(starting_time, ': Program started.')


    # ==================== Pull Stock Data ====================
    # stock_puller = StockPuller(api_key=GlobalConfig.ALPHA_VANTAGE_API_KEY_EXTENDED_HISTORY,
    #                            ticker=GlobalConfig.TESLA_TICKER_STR, interval=GlobalConfig.FIVE_MIN_INTERVAL)
    # stock_puller.pull_data()

    # ==================== Parse Stock Data ====================
    stock_parser = StockParser(ticker=GlobalConfig.TESLA_TICKER_STR, interval=GlobalConfig.ONE_MIN_INTERVAL)
    rec_list = stock_parser.parse_stock_data()


    # ==================== Visualize Stock Data ====================
    stock_visualizer = StockVisualizer(single_stock_recording_list=rec_list)
    stock_visualizer.plot_all_in_one_chart()

    # ending time
    ending_time=datetime.now()
    print(ending_time, ': Program finished.')

    # execution length
    print('Program took:', ending_time-starting_time, 'to run.')