import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# membuat fungsi untuk melakukan prediksi
def predict():
    # membaca dataset
    df = pd.read_csv('IHSG.csv')

    # mengubah format tanggal menjadi datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # mengubah tipe data kolom 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume' menjadi numerik
    df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].apply(pd.to_numeric, errors='coerce')

    # menghapus baris yang memiliki nilai null
    df.dropna(inplace=True)

    # membagi dataset menjadi data latih dan data uji
    train_df = df[df['Date'] < '2019-01-01']
    test_df = df[df['Date'] >= '2019-01-01']

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

    # memuat model
    model = load_model('best_model.h5')

    # melakukan prediksi
    test_x = test_x.astype('float32')
    test_pred = model.predict(test_x)

    # menghitung akurasi dan rmse
    test_rmse = np.sqrt(np.mean(np.square(test_pred - test_y)))
    accuracy_label.config(text='Akurasi (RMSE) pada data uji: {:.4f}'.format(test_rmse))

    # menampilkan grafik harga saham aktual dan prediksi
    fig = plt.figure(figsize=(8,6))
    plt.plot(test_df['Date'], test_y * (train_max['Close'] - train_min['Close']) + train_min['Close'], label='Actual')
    plt.plot(test_df['Date'], test_pred * (train_max['Close'] - train_min['Close']) + train_min['Close'], label='Prediction')
    plt.title('Prediksi Harga Saham')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga Saham (dalam Rupiah)')
    plt.legend()
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0)

# membuat tampilan GUI
root = tk.Tk()
root.title('Prediksi Harga Saham')
root.geometry('800x600')

# membuat frame untuk menampilkan grafik
canvas_frame = tk.Frame(root, bg='white', bd=2, relief=tk.SUNKEN)
canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# membuat tombol untuk melakukan prediksi
predict_button = tk.Button(root, text='Prediksi', command=predict)
predict_button.pack(side=tk.TOP, pady=10)

#membuat label untuk menampilkan akurasi
accuracy_label = tk.Label(root, text='Akurasi (RMSE) pada data uji: -', font=('Helvetica', 12))
accuracy_label.pack(side=tk.BOTTOM, pady=10)

root.mainloop()