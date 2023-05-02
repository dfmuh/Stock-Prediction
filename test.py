import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# membaca dataset
df = pd.read_csv('IHSG.csv')

# mengubah format tanggal menjadi datetime
df['Date'] = pd.to_datetime(df['Date'])

# mengubah tipe data kolom 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume' menjadi numerik
df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].apply(pd.to_numeric, errors='coerce')

# menghapus baris yang memiliki nilai null
df.dropna(inplace=True)

# normalisasi data
min_val = df.min()
max_val = df.max()
df = (df - min_val) / (max_val - min_val)

# memisahkan fitur dan label
x = df.drop(['Close'], axis=1).values.astype('float32')
y = df['Close'].values.astype('float32')

# memuat model
model = load_model('best_model.h5')

# melakukan prediksi
y_pred = model.predict(x)

# mengembalikan data ke skala aslinya
df = df * (max_val - min_val) + min_val
y = y * (max_val['Close'] - min_val['Close']) + min_val['Close']
y_pred = y_pred * (max_val['Close'] - min_val['Close']) + min_val['Close']

# menampilkan grafik aktual dan prediksi
plt.plot(df['Date'], y, label='Aktual')
plt.plot(df['Date'], y_pred, label='Prediksi')
plt.legend()
plt.show()
