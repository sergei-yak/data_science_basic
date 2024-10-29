# data_science_basic
data science simple group project

# Objective: Apply data science concepts on a dataset of your choice.

# Tasks:
    # Acquire, clean, and preprocess data.
    # Perform EDA and visualize key insights.
	# Build and evaluate a machine learning model.

# Requirements:
    # Work on this as a group (Same team as the previous GIT exercise).
    # Use a dataset that is not used in the class.
    # Use at least 3 different visualization techniques. - what techniques?
    # Use at least 1 different machine learning algorithms.
    # Use at least 2 different evaluation metrics.
    # Use at least 2 different preprocessing techniques.

# Submission Timeline:
    # Submit the code and a report in 3 weeks.
    # The report should include:
        # Introduction to the dataset.
        # Data cleaning and preprocessing steps.
        # EDA and key insights.
        # Machine learning model building and evaluation.
        # Conclusion.
        # References (if any).

# Intro to dataset
	The input data was taken from Binance exchange API. It is BTC_USDT futures with columns: Event time,min_a_price,max_b_price,a_quantity,b_quantity,volume_imbalance,mid_price,micro_price,Kline start time,Kline close time,Interval,Open price,Close price,High price,Low price,Base asset volume,Number of trades,Quote asset volume,Taker buy base asset volume,Taker buy quote asset volume.
	
	For simplicity of this project we just use columns: Event time, Open price, Close price, High price, Low price.
 	
  	Time frame is in seconds from 2024-05-19 22:44:02.997 to 2024-05-20 02:24:30.632.

# Data claning and preprocessing
	preprocessing techniques we used:
 	df_binance = df_binance.dropna()
	df_binance = df_binance.reset_index(drop=True)
	df_binance['Event time'] = pd.to_datetime(df_binance['Event time'])
	df_short = df_binance[['Event time', 'Open price', 'High price', 'Low price', 'Close price']]
	df_short = df_short.sort_values(by='Event time')

# EDA
Summary Statistics:


# Machine learning
In the project we used two models:
- ElasticNet() - linear regressiion...
- LSTM model - nearul network ....

# Conclusions
....


# Output
<img width="1507" alt="Screenshot 2024-10-28 at 8 04 11 PM" src="https://github.com/user-attachments/assets/93d62aaf-5e50-4700-8257-115562f78a7a">
