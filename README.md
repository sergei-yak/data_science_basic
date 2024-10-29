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

# Introduction to Dataset
- The input data was sourced from the Binance Exchange API.
- It consists of BTC_USDT futures data with columns: Event Time, Min A Price, Max B Price, A Quantity, B Quantity, Volume Imbalance, Mid Price, Micro Price, Kline Start Time, Kline Close Time, Interval, Open Price, Close Price, High Price, Low Price, Base Asset Volume, Number of Trades, Quote Asset Volume, Taker Buy Base Asset Volume, and Taker Buy Quote Asset Volume.
- For simplicity in this project, we used the following columns: Event Time, Open Price, Close Price, High Price, and Low Price.
- The time frame is in seconds, ranging from 2024-05-19 22:44:02.997 to 2024-05-20 02:24:30.632.

# Data cleaning and preprocessing
1. We dropped NA values: df_binance = df_binance.dropna()
2. We reset the index of the dataframe: df_binance = df_binance.reset_index(drop=True)
3. We converted the 'Event Time' column to datetime format: df_binance['Event time'] = pd.to_datetime(df_binance['Event time'])
4. We selected specific columns: df_short = df_binance[['Event time', 'Open price', 'High price', 'Low price', 'Close price']]
5. We sorted the values by the 'Event Time' column: df_short = df_short.sort_values(by='Event time')

# Exploratory Data Analysis (EDA)
	Summary Statistics:
	We used the describe() method to gain key insights:
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
In this project, we used two models to compare their performance:

ElasticNet - A linear regression model used for regression tasks (using the scikit-learn library).
LSTM Model - A recurrent neural network that excels at capturing long-term dependencies (using the PyTorch library). For better performance we use biderectional LSTM model.

Evaluation Metrics Used:

- MAE - Mean Absolute Error, which measures the average absolute difference between predicted and actual values. Lower MAE indicates better performance.
- R² Score - Measures the proportion of variance in the target variable that is explained by the model. Higher R² scores indicate a better fit.
- MAPE - Mean Absolute Percentage Error, which measures the average percentage difference between predicted and actual values. Lower MAPE indicates better accuracy.

# Conclusions
- The Bidirectional LSTM model outperforms the ElasticNet model across all key metrics: MAE, R² Score, and MAPE.
- The ElasticNet model showed high average error and low explanatory power (as indicated by the low R² Score), making it less reliable for predicting Bitcoin prices.
- The Bidirectional LSTM model achieved lower average errors, higher variance explanation, and minimal percentage-based errors. This suggests that it captures the nonlinear patterns and temporal dependencies in Bitcoin prices more effectively than the ElasticNet model.

# References
- Binance API - https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams
- Plotly library for charts - https://plotly.com/python/subplots/

# Output (was created using plotly library)
<img width="1507" alt="Screenshot 2024-10-28 at 8 04 11 PM" src="https://github.com/user-attachments/assets/93d62aaf-5e50-4700-8257-115562f78a7a">
