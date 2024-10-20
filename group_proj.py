# Load a dataset
from sklearn.datasets import load_wine, load_diabetes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#create dataframe pandas from csv file:
df_binance = pd.read_csv('df_combined.csv') # this data is from binance exchange API btc to usdt price data

#preprocessing techniques
df_binance = df_binance.dropna()
df_binance = df_binance.reset_index(drop=True)
df_binance['Event time'] = pd.to_datetime(df_binance['Event time'])

# EDA and key insights
print("Summary Statistics:")
print(df_binance.describe())
print("Correlation Matrix:")
#print(df_binance.corr())
print(df_binance.columns)
print(df_binance.info())


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
X_train, X_test, y_train, y_test = train_test_split(df_binance[['mid_price', 'Open price', 'High price', 'Low price']], df_binance[['Close price']], test_size=0.2)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Output the Test Data
print("Test Data Set:")
print(X_test)

# Make predictions
predictions = model.predict(X_test)

# make evaluation metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Print evaluation metrics
print("Mean Squared Error: ", mean_squared_error(y_test, predictions))
print("Mean Absolute Error: ", mean_absolute_error(y_test, predictions))
print("R2 Score: ", r2_score(y_test, predictions))


# Print predictions
print("Predictions on test data:")
print(predictions)
#build a line chart in plotly for predictions
import plotly.express as px
fig = px.line(predictions)
fig.show()
