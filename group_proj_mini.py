import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from sklearn.linear_model import ElasticNet, LinearRegression
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#create dataframe pandas from csv file:
df_binance = pd.read_csv('df_combined.csv') # this data is from binance exchange API btc to usdt price data

#preprocessing techniques
df_binance = df_binance.dropna()
df_binance = df_binance.reset_index(drop=True)
df_binance['Event time'] = pd.to_datetime(df_binance['Event time'])
df_short = df_binance[['Event time', 'Open price', 'High price', 'Low price', 'Close price']]
df_short = df_short.sort_values(by='Event time')

# EDA and key insights
print("Summary Statistics:")
print(df_short.describe())
print(df_short.columns)
print(df_short.info())

# Summary statistics for each price type
price_types = ['Open', 'High', 'Low', 'Close']
mean_prices = [66917.51, 66929.86, 66904.66, 66917.04]
min_prices = [66323.60, 66358.64, 66279.64, 66289.40]
max_prices = [67278.69, 67278.23, 67277.10, 67289.48]

# Bar width
bar_width = 0.25

# Positions for each bar on the x-axis
r1 = np.arange(len(price_types))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plotting bars
plt.figure(figsize=(10, 6))
plt.bar(r1, mean_prices, color='b', width=bar_width, edgecolor='grey', label='Mean')
plt.bar(r2, min_prices, color='g', width=bar_width, edgecolor='grey', label='Min')
plt.bar(r3, max_prices, color='r', width=bar_width, edgecolor='grey', label='Max')

# Adding labels and title
plt.xlabel('Price Type', fontweight='bold')
plt.ylabel('Price', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(price_types))], price_types)
plt.title('Summary Statistics for Open, High, Low, and Close Prices')
# Correlation matrix excluding 'Event Time' column
correlation_matrix = df_short.drop(columns=['Event time']).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f",
            cmap='coolwarm', square=True, cbar=True)
plt.title('Correlation Heatmap')
plt.show()

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
model = ElasticNet(random_state=42)
#model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions = model.predict(X_test_scaled)

# Output predictions and actual values for comparison
predictions_df = pd.DataFrame({
    'Event time': event_time_test.reset_index(drop=True), 
    'Predicted': predictions,
    'Actual': y_test.reset_index(drop=True),
    'Model': 'ElasticNet'
})
predictions_df.reset_index(drop=True, inplace=True)
print(predictions_df.sort_values(by='Event time'))

# make evaluation metrics

# Print evaluation metrics
print("Mean Squared Error: ", mean_squared_error(
           predictions_df['Actual'], predictions_df['Predicted']))
print("Mean Absolute Error: ", mean_absolute_error(
            predictions_df['Actual'], predictions_df['Predicted']))
print("R2 Score: ", r2_score(
            predictions_df['Actual'], predictions_df['Predicted']))
print("Mean Absolute Percentage Error: ", mean_absolute_error(
            predictions_df['Actual'], predictions_df['Predicted'])/predictions_df['Actual'].mean())

#create df of evaluation metrics
df_evaluation = pd.DataFrame({
    'Mean Squared Error': [round(mean_squared_error(predictions_df['Actual'], predictions_df['Predicted']), 4)],        
    'Mean Absolute Error': [round(mean_absolute_error(predictions_df['Actual'], predictions_df['Predicted']),4)],
    'R2 Score': [round(r2_score(predictions_df['Actual'], predictions_df['Predicted']),4)],
    'Mean Absolute Percentage Error': [round((mean_absolute_error(predictions_df['Actual'], predictions_df['Predicted'])/predictions_df['Actual'].mean()),4)],
    'Model': ['ElasticNet']
})


# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df_short[['High price', 'Low price', 'Open price',  'Close price']])

X, y = [], []
time_steps = 60
for i in range(time_steps, len(df_scaled)):
    X.append(df_scaled[i - time_steps:i, :])
    y.append(df_scaled[i, 3]) #for y we use just Close price column

X, y = np.array(X), np.array(y)
#X = X.reshape((X.shape[0], X.shape[1], 1))

# train-test split
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
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_size, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        h_0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc1(out[:, -1, :])  # Take the last time step's output
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# Hyperparameters
input_size = 4 #because we have 4 features
hidden_size = 100
num_layers = 3
output_size = 1
learning_rate = 0.001
num_epochs = 5
batch_size = 64
dropout_rate = 0.2

# Instantiate the model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate)
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Evaluation and Predictions
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

# Convert predictions and true values back to the original scale

y_pred_rescaled = scaler.inverse_transform(np.concatenate(
            (np.zeros((y_pred.shape[0], 3)), y_pred.numpy()), axis=1))[:,3]

y_test_rescaled = scaler.inverse_transform(np.concatenate(
            (np.zeros((y_test.shape[0], 3)), y_test.numpy()), axis=1))[:,3]

#add new data from LSTM model to predictions_df for comparison
predictions_df_lstm = pd.DataFrame({
    'Event time': df_short['Event time'].values[-len(y_pred_rescaled):].ravel(),  # Flatten if needed
    'Predicted': y_pred_rescaled.ravel(),  # Flatten if needed
    'Actual': y_test_rescaled.ravel(),  # Flatten if needed
    'Model': 'LSTM bidirectional'
})

# Reset index and concatenate
predictions_df_lstm.reset_index(drop=True, inplace=True)
predictions_df = pd.merge(predictions_df, predictions_df_lstm, on='Event time', how='inner')

print(predictions_df.columns)


# Evaluation metrics
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
mape = mae / y_test_rescaled.mean()

print("Mean Squared Error: ", mse)
print("Mean Absolute Error: ", mae)
print("R2 Score: ", r2)
print("Mean Absolute Percentage Error: ", mape)

# add new evaluation metrics to the df
df_evaluation.loc[1] = [round(mae,4), round(r2,4), round(mape,4), 'LSTM bidirectional']

#create dashboard
from plotly.subplots import make_subplots
fig = make_subplots(rows=2, cols=2, subplot_titles=("Price of BTC futures", "Close Price Distribution of BTC futures",
                                    "Evaluation Metrics", "Predicted vs Actual Prices"),
                                    horizontal_spacing=0.1,
                                    specs=[[{"type": "xy"}, {"type": "xy"}],[{"type": "domain"}, {"type": "xy"}],
    ] )

fig.add_trace(go.Candlestick(x=df_short['Event time'],
                             open=df_short['Open price'],
                             high=df_short['High price'],
                             low=df_short['Low price'],
                             close=df_short['Close price'],
                             name='Price of BTC futures'),  row=1, col=1)
fig.add_trace(go.Histogram(x=df_short['Close price'], name='Price Distribution'), row=1, col=2)
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
fig.add_trace(go.Scatter(x=predictions_df[predictions_df['Model_y'] == "LSTM bidirectional"].tail(60).sort_values(by='Event time')['Event time'], y=predictions_df.tail(60).sort_values(by='Event time')['Predicted_y'], mode='lines', name='Prediction LSTM'), row=2, col=2)
fig.add_trace(go.Scatter(x=predictions_df[predictions_df['Model_y'] == "LSTM bidirectional"].tail(60).sort_values(by='Event time')['Event time'], y=predictions_df.tail(60).sort_values(by='Event time')['Actual_y'], mode='lines', name='Actual price'), row=2, col=2)
fig.add_trace(go.Scatter(x=predictions_df[predictions_df['Model_x'] == "ElasticNet"].tail(60).sort_values(by='Event time')['Event time'], y=predictions_df.tail(60).sort_values(by='Event time')['Predicted_x'], mode='lines', name='Prediction ElesticNet'), row=2, col=2)


#add names for subplots x/y axis
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=1, col=2)
fig.update_yaxes(title_text="Price", row=2, col=2)
fig.update_xaxes(title_text="Time (data in seconds)", row=1, col=1)
fig.update_xaxes(title_text="Price", row=1, col=2)
fig.update_xaxes(title_text="Time (data in seconds)", row=2, col=2)
fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
fig.update_layout(height=900, width=1500, title_text="Time Series Forecasting Dashboard", title_x=0.5)

# Adding an annotation for the evaluation metrics conclusions
fig.add_annotation(
    text='<b>MAE</b> - the average absolute difference between predicted and actual values. '
         '<br>A lower MAE indicates better performance. '
         '<br>'
         '<br><b>R² score</b> - the proportion of variance in the target variable that is explained by the model. '
         '<br>A higher R² score indicates a better fit. '
         '<br>'
         '<br> <b>MAPE</b> - the average percentage difference between predicted and actual values. '
         '<br>A lower MAPE indicates better accuracy. ',
    x=-0.005,
    y=0.00,
    xref="paper",
    yref="paper",
    showarrow=False,
    arrowhead=2,
    bgcolor='rgba(255, 255, 255, 0.8)',
    bordercolor='#19D3F3',
    borderwidth=2
)

# Adding a legend
plt.legend()

# Display the chart
# Summary statistics for each price type
price_types = ['Open', 'High', 'Low', 'Close']
mean_prices = [66917.51, 66929.86, 66904.66, 66917.04]
min_prices = [66323.60, 66358.64, 66279.64, 66289.40]
max_prices = [67278.69, 67278.23, 67277.10, 67289.48]

# Bar width
bar_width = 0.25

# Positions for each bar on the x-axis
r1 = np.arange(len(price_types))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plotting bars
plt.figure(figsize=(10, 6))
plt.bar(r1, mean_prices, color='b', width=bar_width, edgecolor='grey', label='Mean')
plt.bar(r2, min_prices, color='g', width=bar_width, edgecolor='grey', label='Min')
plt.bar(r3, max_prices, color='r', width=bar_width, edgecolor='grey', label='Max')

# Adding labels and title
plt.xlabel('Price Type', fontweight='bold')
plt.ylabel('Price', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(price_types))], price_types)
plt.title('Summary Statistics for Open, High, Low, and Close Prices')

# Adding a legend
plt.legend()

plt.show()

fig.write_html('df_dashboard.html')
fig.show()
