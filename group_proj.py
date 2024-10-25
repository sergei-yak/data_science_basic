# Load a dataset
from sklearn.datasets import load_wine, load_diabetes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

#create dataframe pandas from csv file:
df_binance = pd.read_csv('data_science_basic/df_combined.csv') # this data is from binance exchange API btc to usdt price data

#preprocessing techniques
df_binance = df_binance.dropna()
df_binance = df_binance.reset_index(drop=True)
df_binance['Event time'] = pd.to_datetime(df_binance['Event time'])
df_short = df_binance[['Event time', 'Open price', 'High price', 'Low price', 'Close price']]
df_short = df_short.sort_values(by='Event time')

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
X_train, X_test, y_train, y_test, event_time_train, event_time_test = train_test_split(
    df_short[['Open price', 'High price', 'Low price']],
    df_short['Close price'], 
    df_short['Event time'],
    test_size=0.2,
    random_state=42
)

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
    'Event time': event_time_test.reset_index(drop=True), 
    'Predicted': predictions,
    'Actual': y_test.reset_index(drop=True) 
})
predictions_df.reset_index(drop=True, inplace=True)
print(predictions_df.sort_values(by='Event time'))

# make evaluation metrics
#from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Print evaluation metrics
#print("Mean Squared Error: ", mean_squared_error(predictions_df['Actual'], predictions_df['Predicted']))
print("Mean Absolute Error: ", mean_absolute_error(predictions_df['Actual'], predictions_df['Predicted']))
print("R2 Score: ", r2_score(predictions_df['Actual'], predictions_df['Predicted']))
print("Mean Absolute Percentage Error: ", mean_absolute_error(predictions_df['Actual'], predictions_df['Predicted'])/predictions_df['Actual'].mean())

#create df of evaluation metrics
df_evaluation = pd.DataFrame({
    'Mean Absolute Error': [round(mean_absolute_error(predictions_df['Actual'], predictions_df['Predicted']),4)],
    'R2 Score': [round(r2_score(predictions_df['Actual'], predictions_df['Predicted']),4)],
    'Mean Absolute Percentage Error': [round((mean_absolute_error(predictions_df['Actual'], predictions_df['Predicted'])/predictions_df['Actual'].mean()),4)]
})

#build a line chart in plotly for predictions
import plotly.express as px

#show the predictions for last 1 minute
last_60_values = predictions_df.tail(60)
# Reshape the data using melt to combine 'Predicted' and 'Actual' into a single column
last_60_values_melted = last_60_values.melt(id_vars=['Event time'], value_vars=['Predicted', 'Actual'],
                                            var_name='Type', value_name='Price')
last_60_values_melted = last_60_values_melted.sort_values(by='Event time')
# Create a line chart with color differentiation between Predicted and Actual
fig = px.line(last_60_values_melted, x="Event time", y="Price", color='Type', title='Predicted vs Actual Prices')
fig.show()

#create dashboard
from plotly.subplots import make_subplots
fig = make_subplots(rows=2, cols=2, subplot_titles=("Price of BTC futures", "Close Price Distribution of BTC futures", 
                                    "Evaluation Metrics", "Predicted vs Actual Prices"),
                                    specs=[[{"type": "xy"}, {"type": "xy"}],[{"type": "domain"}, {"type": "xy"}]
    ] )

fig.add_trace(go.Candlestick(x=df_short['Event time'],
                             open=df_short['Open price'],
                             high=df_short['High price'],
                             low=df_short['Low price'],
                             close=df_short['Close price'],
                             name='Price of BTC futures'),  row=1, col=1)
fig.add_trace(go.Histogram(x=df_short['Close price'], name='Close Price Distribution'), row=1, col=2)
#fig.add_trace(go.Bar(x=['Mean Absolute Error', 'R2 Score', 'Mean Absolute Percentage Error'], y=[mean_absolute_error(predictions_df['Actual'], predictions_df['Predicted']),r2_score(predictions_df['Actual'], predictions_df['Predicted']), mean_absolute_error(predictions_df['Actual'], predictions_df['Predicted'])/predictions_df['Actual'].mean()], name='Evaluation metrics'), row=2, col=1)

fig.add_trace(go.Table(
        header=dict(
            values=list(df_evaluation.columns),
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[df_evaluation[col] for col in df_evaluation.columns],
            fill_color='lavender',
            align='left'
        )
    ),
    row=2, col=1
)
fig.add_trace(go.Scatter(x=predictions_df.tail(60).sort_values(by='Event time')['Event time'], y=predictions_df.tail(60).sort_values(by='Event time')['Predicted'], mode='lines', name='Predicted price'), row=2, col=2)
fig.add_trace(go.Scatter(x=predictions_df.tail(60).sort_values(by='Event time')['Event time'], y=predictions_df.tail(60).sort_values(by='Event time')['Actual'], mode='lines', name='Actual price'), row=2, col=2)

#add names for subplots x/y axis
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Price", row=1, col=2)
#fig.update_yaxes(title_text="Price", row=2, col=1)
fig.update_yaxes(title_text="Price", row=2, col=2)
fig.update_xaxes(title_text="Time", row=1, col=1)
fig.update_xaxes(title_text="Time", row=1, col=2)
#fig.update_xaxes(title_text="Time", row=2, col=1)
fig.update_xaxes(title_text="Time", row=2, col=2)
fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
fig.update_layout(height=1000, width=1200, title_text="Time Series Forecasting Dashboard", title_x=0.5)

fig.write_html('df_dashboard.html')
fig.show()

'''
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df_short[['Close price']])

X, y = [], []
time_steps = 60
for i in range(time_steps, len(df_scaled)):
    X.append(df_scaled[i - time_steps:i, 0])
    y.append(df_scaled[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

    
# Splitting into training and testing sets
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Building and training the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Making predictions on the test set
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluation metrics
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)

print("Mean Absolute Error: ", mae)
print("R2 Score: ", r2_score(y_test_rescaled, y_pred_rescaled))
print("Mean Absolute Percentage Error: ", mean_absolute_error(y_test_rescaled, y_pred_rescaled)/y_test_rescaled.mean())
'''
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df_short[['Close price']])

X, y = [], []
time_steps = 60
for i in range(time_steps, len(df_scaled)):
    X.append(df_scaled[i - time_steps:i, 0])
    y.append(df_scaled[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Splitting into training and testing sets
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 50)  # hidden state (num_layers, batch_size, hidden_size)
        c_0 = torch.zeros(2, x.size(0), 50)  # cell state (num_layers, batch_size, hidden_size)

        # LSTM forward pass
        out, _ = self.lstm(x, (h_0, c_0))  # Pass the input through the LSTM layers
        out = self.fc(out[:, -1, :])        # Pass the last output through the dense layer
        return out

# Hyperparameters
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1
learning_rate = 0.001
num_epochs = 20
batch_size = 32

# Instantiate the model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation and Predictions
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

# Convert predictions and true values back to the original scale
y_pred_rescaled = scaler.inverse_transform(y_pred.numpy())
y_test_rescaled = scaler.inverse_transform(y_test.numpy())

# Evaluation metrics
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
mape = mae / y_test_rescaled.mean()

print("Mean Squared Error: ", mse)
print("Mean Absolute Error: ", mae)
print("R2 Score: ", r2)
print("Mean Absolute Percentage Error: ", mape)