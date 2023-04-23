# Import necessary libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('sales_data.csv')

# Aggregate sales data at monthly level
monthly_data = data.groupby(['item_name', 'month_year'])['sales_per_month'].sum().reset_index()

# Prepare the data for LSTM input
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix >= len(data):
            break
        seq_x, seq_y = data[i:end_ix, :-1], data[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(monthly_data[['sales_per_month']])

# Split the data into training and test sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size, :]
test_data = scaled_data[train_size:, :]

# Prepare the data for LSTM input with a time step of 12 months
n_steps = 12
X_train, y_train = prepare_data(train_data, n_steps)
X_test, y_test = prepare_data(test_data, n_steps)

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(n_steps, 1)))
model.add(Dropout(0.2))
model.add(LSTM(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the LSTM model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=2, validation_data=(X_test, y_test))

# Make predictions for the next 12 months for each item_name
preds = []
for item in monthly_data['item_name'].unique():
    item_data = monthly_data[monthly_data['item_name'] == item]['sales_per_month'].values
    item_data = scaler.transform(item_data.reshape(-1, 1))
    X_item, _ = prepare_data(item_data)
