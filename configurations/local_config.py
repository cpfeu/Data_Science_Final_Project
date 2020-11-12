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
