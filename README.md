###requirements.txt
This file holds all packages that need to be installed 
with the respective version sot that all functions are
executable. We recommend creating a new virtual 
environment for that.

###local_config.py
This file contains variables that are individual to every 
machine. The directories to which the paths reference
should be manually created before the program is run.  

###global_config.py
This file holds all variables from *local_config.py* 
and global variables which are referenced all over the project.

###main_NAME.py
These files holds all execution commands. 

###stock_pulling.py
This file contains code with which stock data sets can be 
downloaded with an Alpha Vantage API and stored in a csv file.

###stock_preprocessing.py
A file to parse the stock data sets and perform a number
of preprocessing steps for visualization and 
analysis purposes.

###stock_visualization.py
A visualization file to plot various features of the 
stock data sets