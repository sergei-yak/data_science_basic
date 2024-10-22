# Load a dataset
from sklearn.datasets import load_wine, load_diabetes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

#create dataframe pandas from csv file:
df_binance = pd.read_csv('df_combined.csv') # this data is from binance exchange API btc to usdt price data

#preprocessing techniques
df_binance = df_binance.dropna()
df_binance = df_binance.reset_index(drop=True)
df_binance['Event time'] = pd.to_datetime(df_binance['Event time'])
df_short = df_binance[['Open price', 'High price', 'Low price', 'Close price']]

# EDA and key insights
print("Summary Statistics:")
print(df_short.describe())
#print("Correlation Matrix:")
#print(df_binance.corr())
print(df_short.columns)
print(df_short.info())


#build candlestick chart in plotly
import plotly.graph_objects as go
fig = go.Figure(data=[go.Candlestick(x=df_binance['Event time'],
                open=df_binance['Open price'],
                high=df_binance['High price'],
                low=df_binance['Low price'],
                close=df_binance['Close price'])])
#fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()

#build a line chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_binance['Event time'], y=df_binance['Open price'], mode='lines', name='Open price'))
fig.add_trace(go.Scatter(x=df_binance['Event time'], y=df_binance['High price'], mode='lines', name='High price'))
fig.add_trace(go.Scatter(x=df_binance['Event time'], y=df_binance['Low price'], mode='lines', name='Low price'))
fig.add_trace(go.Scatter(x=df_binance['Event time'], y=df_binance['Close price'], mode='lines', name='Close price'))
fig.show()

#plot histogram of one feature
df_binance['Close price'].plot(kind='hist', title='Close Price Distribution')
plt.show()

# Split the data into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(df_short[['Open price', 'High price', 'Low price']], df_short[['Close price']], test_size=0.2)

print('test')
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_short[['Open price', 'High price', 'Low price']],
                                                    df_short['Close price'], test_size=0.2, random_state=42)

# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the ElasticNet model
from sklearn.linear_model import ElasticNet
model = ElasticNet(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions = model.predict(X_test_scaled)

# Output predictions and actual values for comparison
predictions_df = pd.DataFrame({
    'Predicted': predictions,
    'Actual': y_test
})
predictions_df.reset_index(drop=True, inplace=True)
print(predictions_df)

# make evaluation metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Print evaluation metrics
print("Mean Squared Error: ", mean_squared_error(predictions_df['Actual'], predictions_df['Predicted']))
print("Mean Absolute Error: ", mean_absolute_error(predictions_df['Actual'], predictions_df['Predicted']))
print("R2 Score: ", r2_score(predictions_df['Actual'], predictions_df['Predicted']))
print("Mean Absolute Percentage Error: ", mean_absolute_error(predictions_df['Actual'], predictions_df['Predicted'])/predictions_df['Actual'].mean())

#build a line chart in plotly for predictions
import plotly.express as px
fig = px.line(predictions_df['Predicted'])
fig.show()
