import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv('IHSG.csv')

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort dataset by date
df = df.sort_values('Date')

# Set Date column as index
df.set_index('Date', inplace=True)

# Create dataframe with only the Close column
data = df.filter(['Close'])

# Convert dataframe to numpy array
dataset = data.values

# Get number of rows to train the model on (80% of the data)
training_data_len = int(np.ceil(0.8 * len(dataset)))

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Load the trained model
model = load_model('best_model.h5')

# Create the testing data set
test_data = scaled_data[training_data_len-60:, :]

# Split the data into x_test and y_test sets
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the x_test to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plot the actual and predicted prices
plt.plot(y_test, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
