import os


class LocalConfig:

    # CPU count
    NUM_CPUs = os.cpu_count() - 2

    # Stock Pulling API
    ALPHA_VANTAGE_API_KEY_EXTENDED_HISTORY = 'OP5D9A64ADCHRG8R'

    # Data base path
    DATA_BASE_PATH = 'C:/Users/cpfeu/Data/Data_Science_Final_Project'

    # Programming outputs base path
    PROGRAMMING_OUTPUTS_BASE_PATH = 'C:/Users/cpfeu/Documents/Cornell/Courses/' \
                                    'Data_Science_in_the_Wild/Project/Programming_Outputs'

    # Isha Twitter keys
    access_token = "1325482462038855680-5CeYdBq8aacwxbdFZ6ta0AzaMgJpa1"
    access_token_secret = "PK6ZReEZwbfqmF6w9t7GDxdFI7d1EP5RR9lmLwZjRbR8q"
    consumer_key = "xQAPcERrIPxKM63WUIJSsKy3u"
    consumer_secret = "pv2JSX65cDe0DiWVeejDBpesWD05uhxxH5OaMGcXAEsRL0kiRm"