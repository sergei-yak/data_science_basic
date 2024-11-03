# Group mini project - data_science_basic

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
- Summary Statistics:
  		
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
- Info to get information about columns data type using info() method:
  		
		Data columns (total 5 columns):
		 #   Column       Non-Null Count  Dtype
		---  ------       --------------  -----
		 0   Event time   5297 non-null   datetime64[ns]
		 1   Open price   5297 non-null   float64
		 2   High price   5297 non-null   float64
		 3   Low price    5297 non-null   float64
		 4   Close price  5297 non-null   float64

# Machine learning
In this project, we used two models to compare their performance:

- ElasticNet - A linear regression model used for regression tasks (using the scikit-learn library).
- LSTM Model - A recurrent neural network that excels at capturing long-term dependencies (using the PyTorch library). For better performance we use biderectional LSTM model. 

Evaluation Metrics Used:

- MSE - Mean Squared Error, which measures the average of the squared differences between predicted and actual values. Lower MSE indicates better performance.
- MAE - Mean Absolute Error, which measures the average absolute difference between predicted and actual values. Lower MAE indicates better performance.
- R² Score - Measures the proportion of variance in the target variable that is explained by the model. Higher R² scores indicate a better fit.
- MAPE - Mean Absolute Percentage Error, which measures the average percentage difference between predicted and actual values. Lower MAPE indicates better accuracy.

# Conclusions
- The Bidirectional LSTM model outperforms the ElasticNet model across all key metrics: MAE, R² Score, and MAPE.
- The ElasticNet model showed high average error and low explanatory power (as indicated by the low R² Score), making it less reliable for predicting Bitcoin prices.
- The Bidirectional LSTM model achieved lower average errors, higher variance explanation, and minimal percentage-based errors. This suggests that it captures the nonlinear patterns and temporal dependencies in Bitcoin prices more effectively than the ElasticNet model.

# References
- Explanation of LSTM model - https://medium.com/@anishnama20/understanding-lstm-architecture-pros-and-cons-and-implementation-3e0cca194094](https://medium.com/towards-data-science/lstm-for-google-stock-price-prediction-e35f5cc84165)
- Binance API - https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams
- Plotly library for charts - https://plotly.com/python/subplots/

# Output (was created using plotly library)
<img width="1492" alt="Screenshot 2024-10-29 at 12 49 32 AM" src="https://github.com/user-attachments/assets/911392a2-1450-4f78-b7e6-4633245cdbe8">

