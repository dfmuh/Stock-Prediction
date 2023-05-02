import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# membaca dataset
df = pd.read_csv('IHSG.csv')

# mengubah format tanggal menjadi datetime
df['Date'] = pd.to_datetime(df['Date'])

# mengubah tipe data kolom 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume' menjadi numerik
df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].apply(pd.to_numeric, errors='coerce')

# menghapus baris yang memiliki nilai null
df.dropna(inplace=True)

# membagi dataset menjadi data latih dan data uji
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

# normalisasi data latih dan data uji
train_min = train_df.min()
train_max = train_df.max()
train_df = (train_df - train_min) / (train_max - train_min)
test_df = (test_df - train_min) / (train_max - train_min)

# memisahkan fitur dan label
train_x = train_df.drop(['Close'], axis=1).values.astype('float32')
train_y = train_df['Close'].values.astype('float32')
test_x = test_df.drop(['Close'], axis=1).values.astype('float32')
test_y = test_df['Close'].values.astype('float32')

# membuat model
model = Sequential()
model.add(Dense(128, input_dim=train_x.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')

# melatih model
model.fit(train_x, train_y, epochs=100, batch_size=16, verbose=1)

# menyimpan model
model.save('best_model1.h5')

# menghitung akurasi dan rmse
train_pred = model.predict(train_x)
test_pred = model.predict(test_x)
train_rmse = np.sqrt(mean_squared_error(train_y, train_pred))
test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))
print('Akurasi (RMSE) pada data latih:', train_rmse)
print('Akurasi (RMSE) pada data uji:', test_rmse)
