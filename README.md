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
 	1. we droped NA values -> df_binance = df_binance.dropna()
 	2. we reset index of dataframe -> df_binance = df_binance.reset_index(drop=True)
 	3. we changed 'Event time' column datetime format -> df_binance['Event time'] = pd.to_datetime(df_binance['Event time'])
 	4. we chose specific columns from dataframe -> df_short = df_binance[['Event time', 'Open price', 'High price', 'Low price', 'Close price']]
 	5. we sorted values by 'Event time' column -> df_short = df_short.sort_values(by='Event time')

# EDA
	Summary Statistics:
	we used describe() method to show us key insite:
                          Event time    Open price  ...     Low price   Close price
	count                           5297   5297.000000  ...   5297.000000   5297.000000
	mean   2024-05-20 00:35:45.321837824  66917.509166  ...  66904.655182  66917.036058
	min       2024-05-19 22:44:02.997000  66323.600000  ...  66279.640000  66289.400000
	25%    2024-05-19 23:41:33.167000064  66687.820000  ...  66682.000000  66687.920000
	50%    2024-05-20 00:36:13.321999872  67010.010000  ...  66997.940000  67010.000000
	75%    2024-05-20 01:30:52.481999872  67110.190000  ...  67104.040000  67110.180000
	max       2024-05-20 02:24:30.632000  67278.690000  ...  67277.100000  67289.480000
	std                              NaN    225.395426  ...    231.330622    227.803086


# Machine learning
In the project we used two models:
- ElasticNet() - linear regressiion...
- LSTM model - nearul network ....

# Conclusions
The Bidirectional LSTM model outperforms the ElasticNet model on all key metrics: MAE, R² score, and MAPE.
ElasticNet has a high average error and low explanatory power (as shown by the low R² score), making it less reliable for predicting Bitcoin prices.
Bidirectional LSTM achieves lower average errors, higher variance explanation, and minimal percentage-based errors, indicating that it captures the nonlinear patterns and temporal dependencies in Bitcoin prices much more effectively than the ElasticNet.


# Output
<img width="1507" alt="Screenshot 2024-10-28 at 8 04 11 PM" src="https://github.com/user-attachments/assets/93d62aaf-5e50-4700-8257-115562f78a7a">
