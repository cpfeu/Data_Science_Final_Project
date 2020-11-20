###requirements.txt
This file holds all packages that need to be installed 
with the respective version sot that all functions are
executable. We recommend creating a new virtual 
environment for that.

###local_config.py
This file contains variables that are individual to every 
machine. The directories to which the paths reference
should be manually created before the program is run.
The file must be created manually and saved in the 
configurations directory. The file should have the following 
structure:

    import os
    class LocalConfig:

        # CPU count
        NUM_CPUs = os.cpu_count() - 2

        # Stock Pulling API
        ALPHA_VANTAGE_API_KEY_EXTENDED_HISTORY = 'XXX'
    
        # Data base path
        DATA_BASE_PATH = 'XXX'
    
        # Programming outputs base path
        PROGRAMMING_OUTPUTS_BASE_PATH = 'XXXXX'
    
        # Twitter keys
        access_token = "XXX"
        access_token_secret = "XXX"
        consumer_key = "XXX"
        consumer_secret = "XXX" 

###global_config.py
This file holds all variables from *local_config.py* 
and global variables which are referenced all over the project.

###main_NAME.py
These files holds all execution commands. 

###stock_pulling.py
This file contains code with which stock data sets can be 
downloaded with an Alpha Vantage API and stored in a csv file.

###twitter_pulling.py
This file contains code with which historical tweet data 
for the chosen keyword is pulled and stored in a csv file.

###reddit_pulling.py
This file contains code to pull reddit post data for the chosen companies 
and store it in a csv file.

###stock_preprocessing.py
A file to parse the stock data sets and perform a number
of preprocessing steps for visualization and 
analysis purposes.

###stock_visualization.py
A visualization file to plot various features of the 
stock data sets
