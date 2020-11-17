from configurations.local_config import LocalConfig

class GlobalConfig:

    # CPU count
    NUM_CPUs = LocalConfig.NUM_CPUs

    # Stock pulling API
    ALPHA_VANTAGE_API_KEY_EXTENDED_HISTORY = LocalConfig.ALPHA_VANTAGE_API_KEY_EXTENDED_HISTORY

    # Data base path
    DATA_BASE_PATH = LocalConfig.DATA_BASE_PATH

    # Programming outputs base path
    PROGRAMMING_OUTPUTS_BASE_PATH = LocalConfig.PROGRAMMING_OUTPUTS_BASE_PATH

    # Parameters for AlphaVantage API
    SLICE_LIST = ['year1month1', 'year1month2', 'year1month3', 'year1month4', 'year1month5', 'year1month6',
                  'year1month7', 'year1month8', 'year1month9', 'year1month10', 'year1month11', 'year1month12',
                  'year2month1', 'year2month2', 'year2month3', 'year2month4', 'year2month5', 'year2month6',
                  'year2month7', 'year2month8', 'year2month9', 'year2month10', 'year2month11', 'year2month12']

    # Tickers
    TESLA_TICKER_STR = 'TSLA'
    KODAK_TICKER_STR = 'KODK'
    HERTZ_TICKER_STR = 'HTZGQ'
    PALANTIR_TICKER_STR = 'PLTR'
    SNOWFLAKE_TICKER_STR = 'SNOW'
    GOLDMAN_SACHS_TICKER = 'GS'
    BANK_OF_AMERICA_TICKER = 'BAC'

    # Intervals
    ONE_MIN_INTERVAL = '1min'
    FIVE_MIN_INTERVAL = '5min'
    FIFTEEN_MIN_INTERVAL = '15min'
    THIRTY_MIN_INTERVAL = '30min'
    SIXTY_MIN_INTERVAL = '60min'

    # Stock parameters
    STOCK_PARAM_TIME = 'time'
    STOCK_PARAM_OPEN = 'open'
    STOCK_PARAM_CLOSE = 'close'
    STOCK_PARAM_LOW = 'low'
    STOCK_PARAM_HIGH = 'high'
    STOCK_PARAM_VOLUME = 'volume'

    # Plotly colors
    OPEN_COLOR = 'blue'
    CLOSE_COLOR = 'red'
    HIGH_COLOR = 'orange'
    LOW_COLOR = 'green'
    VOLUME_COLOR = 'grey'
