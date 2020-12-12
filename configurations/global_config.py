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

    # Isha Twitter Keys
    ACCESS_TOKEN = LocalConfig.access_token
    ACCESS_TOKEN_SECRET = LocalConfig.access_token_secret
    CONSUMER_KEY = LocalConfig.consumer_key
    CONSUMER_SECRET = LocalConfig.consumer_secret

    # Parameters for AlphaVantage API
    SLICE_LIST = ['year1month1', 'year1month2', 'year1month3', 'year1month4', 'year1month5', 'year1month6',
                  'year1month7', 'year1month8', 'year1month9', 'year1month10', 'year1month11', 'year1month12',
                  'year2month1', 'year2month2', 'year2month3', 'year2month4', 'year2month5', 'year2month6',
                  'year2month7', 'year2month8', 'year2month9', 'year2month10', 'year2month11', 'year2month12']

    # Tickers
    TESLA_TICKER_STR = 'TSLA'
    KODAK_TICKER_STR = 'KODK'
    PALANTIR_TICKER_STR = 'PLTR'
    GOLDMAN_SACHS_TICKER = 'GS'
    BANK_OF_AMERICA_TICKER = 'BAC'
    NIO_TICKER = 'NIO'
    BOEING_TICKER = 'BA'
    NIKOLA_TICKER = 'NKLA'

    TICKER_LIST = [TESLA_TICKER_STR, KODAK_TICKER_STR, PALANTIR_TICKER_STR, GOLDMAN_SACHS_TICKER,
                   BANK_OF_AMERICA_TICKER, NIO_TICKER, BOEING_TICKER, NIKOLA_TICKER]


    # Intervals
    ONE_MIN_INTERVAL = '1min'
    FIVE_MIN_INTERVAL = '5min'
    FIFTEEN_MIN_INTERVAL = '15min'
    THIRTY_MIN_INTERVAL = '30min'
    SIXTY_MIN_INTERVAL = '60min'

    INTERVAL_LIST = [ONE_MIN_INTERVAL, FIVE_MIN_INTERVAL,
                     FIFTEEN_MIN_INTERVAL, THIRTY_MIN_INTERVAL, SIXTY_MIN_INTERVAL]


    # Stock parameters
    STOCK_PARAM_TIME = 'time'
    STOCK_PARAM_OPEN = 'open'
    STOCK_PARAM_CLOSE = 'close'
    STOCK_PARAM_LOW = 'low'
    STOCK_PARAM_HIGH = 'high'
    STOCK_PARAM_VOLUME = 'volume'

    STOCK_PARAM_LIST = [STOCK_PARAM_TIME, STOCK_PARAM_OPEN, STOCK_PARAM_CLOSE,
                        STOCK_PARAM_LOW, STOCK_PARAM_HIGH, STOCK_PARAM_VOLUME]

    # Date string
    DATE_LIST = 'date_list'

    # Stock feature dict parameters
    START_TIME = 'start_time'
    END_TIME = 'end_time'
    SSR_LIST = 'ssr_list'
    OVER_NIGHT_DIFF = 'over_night_difference'
    MAX_MARGIN = 'peak_to_trough'
    ABS_DIFFERENCE = 'open_close_difference'
    PER_CHANGE = 'percentage_change'
    ABS_VOL = 'trading_volume'
    VOL_FLUC = 'volume_stdev'
    PRICE_FLUC = 'price_stdev'


    # Social media dict parameters
    TWEET_LIST = 'tweets'
    PREDICTION_LIST = 'predictions'
    SENTIMENT_LIST = 'sentiments'
    TIMESTAMP_LIST = 'timestamps'
    POSITIVE_PREDICTION_STR = 'positive'
    NEUTRAL_PREDICTION_STR = 'neutral'
    NEGATIVE_PREDICTION_STR = 'negative'
    AVERAGE_SENTIMENT = 'average_sentiment'
    POS_PER_SENTIMENT = 'positive_percentage'
    NEU_PER_SENTIMENT = 'neutral_percentage'
    NEG_PER_SENTIMENT = 'negative_percentage'
    NUM_POS = 'number_posts'
    MAX_SENTIMENT_MARGIN = 'max_sentiment_margin'


    # Twitter feature paramterers
    TWITTER_AVERAGE_SENTIMENT = 'twitter_average_sentiment'
    TWITTER_POS_PER_SENTIMENT = 'twitter_positive_percentage'
    TWITTER_NEU_PER_SENTIMENT = 'twitter_neutral_percentage'
    TWITTER_NEG_PER_SENTIMENT = 'twitter_negative_percentage'
    TWITTER_NUM_POS = 'twitter_number_posts'
    TWITTER_MAX_SENTIMENT_MARGIN = 'twitter_max_sentiment_margin'

    # Reddit feature paramterers
    REDDIT_AVERAGE_SENTIMENT = 'reddit_average_sentiment'
    REDDIT_POS_PER_SENTIMENT = 'reddit_positive_percentage'
    REDDIT_NEU_PER_SENTIMENT = 'reddit_neutral_percentage'
    REDDIT_NEG_PER_SENTIMENT = 'reddit_negative_percentage'
    REDDIT_NUM_POS = 'reddit_number_posts'
    REDDIT_MAX_SENTIMENT_MARGIN = 'reddit_max_sentiment_margin'


    # Time lag directions
    TIME_LAG_RIGHT = 'right'
    TIME_LAG_LEFT = 'left'


    # Plotly colors
    OPEN_COLOR = 'blue'
    CLOSE_COLOR = 'red'
    HIGH_COLOR = 'orange'
    LOW_COLOR = 'green'
    VOLUME_COLOR = 'grey'

