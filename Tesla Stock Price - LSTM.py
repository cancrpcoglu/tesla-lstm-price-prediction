import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')  #hataları yok sayar

import os                           # dosya yolunu bulmamızı sağlar

for dirname,_,filenames in os.walk("YOUR FILE PATH"): #dosyayı bulup içinde gezinir
    for filename in filenames:
        print(os.path.join(dirname, filename))                         #iki yolu birleştirip tek yol oluşturur

data = pd.read_csv('TSLA.csv')                                        # TSLA.csv'sini okur ve açar
length_data = len(data)                                                # toplam satır sayısını hesaplar
split_ratio = 0.7                                                      # %70'i train %30'u validation için kullanılır
length_train = round(length_data * split_ratio)                        # eğitim için ayrılacak veri sayısını ayarlar
length_validation = length_data - length_train                         # doğrulama için kalan veri sayısını atar

train_data = data[:length_train].iloc[:,:2]                             # seçilen satırlardan il 2 sütunu alır [:,:2] tüm satırların il iki sütununu seçer (%70 i alınır)
train_data['Date'] = pd.to_datetime(train_data['Date'])                 # 'train_data' nın tarih ve saat verilerini kolay ve etkili şekilde işlemesini sağlar
validation_data = data[length_train:].iloc[:,:2]                        # Yukarda alınan %70'lik verilerden geri kalan %30'luk kısmı almasını sağlar
validation_data['Date']=pd.to_datetime(validation_data['Date'])         # 'validation_data' nın tatih ve saat verilerini kolay ve etkili şekilde işlemesini sağlar

dataset_train = train_data.Open.values                                  # 'train_data' nın 'Open' valuelarını 'dataset_train' e kaydeder
print(dataset_train.shape)                                              # şeklini düzene sokar
dataset_train = np.reshape(dataset_train, (-1,1))                       # '-1',dizi boyunca oto olarak belirlemesini söyler, '1', her satırda sadece 1 değer olucağını söyler
print(dataset_train.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))                              # MinMaxScaler(feature_range(0,1)): min değer 0, max değer 1 olacak şekilde scale eder
dataset_train_scaled = scaler.fit_transform(dataset_train)              # 'dataset_train'deki verileri 0-1 arasında sıkıştırır: 'fit': min max değeri öğrenir, 'transform' bu değerlere göre ölçekler
dataset_train_scaled.shape

X_train = []
y_train = []
time_step = 50                                                          # Kaç adımda bir yazılacağını gösterir

for i in range(time_step, length_train):
    X_train.append(dataset_train_scaled[i-time_step:i,0])               # Her bir iterasyonda, 'time_step' kadar geçmiş zaman adımını ekler. 'i-time_step:i' aralığı, şu anki zaman adımına kadar olan 'time_step' sayıda geçmiş zaman içerir
    y_train.append(dataset_train_scaled[i,0])                           # Her bir iterasyonda, şu anki zaman adımındaki değeri ekler
X_train, y_train = np.array(X_train), np.array(y_train)
print("Shape of X_train before reshape :",X_train.shape)
print("Shape of y_train before reshape :",y_train.shape)

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))     # Boyut değiştirmeye yarıyor
y_train = np.reshape(y_train,(y_train.shape[0],1))

y_train = scaler.fit_transform(y_train)

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model_lstm = Sequential()
model_lstm.add(
    LSTM(100,return_sequences=True,input_shape = (X_train.shape[1],1))) #64 lstm neuron block
model_lstm.add(
    LSTM(100, return_sequences= False))
model_lstm.add(Dense(64))
model_lstm.add(Dense(1))
model_lstm.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])
history = model_lstm.fit(X_train, y_train, epochs = 300, batch_size =4 )

plt.figure(figsize =(10,5))
plt.plot(history.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("LSTM model, Accuracy vs Epoch")
plt.show()

y_pred = model_lstm.predict(X_train)  # predictions
y_pred = scaler.inverse_transform(y_pred) # scaling back from 0-1 to original
y_pred.shape

y_train = scaler.inverse_transform(y_train) # scaling back from 0-1 to original
y_train.shape

plt.figure(figsize = (30,10))
plt.plot(y_pred, color = "b", label = "y_pred" )
plt.plot(y_train, color = "g", label = "y_train")
plt.xlabel("Days")
plt.ylabel("Open price")
plt.title("LSTM model, Predictions with input X_train vs y_train")
plt.legend()
plt.show()

dataset_validation = validation_data.Open.values  # getting "open" column and converting to array
dataset_validation = np.reshape(dataset_validation, (-1,1))  # converting 1D to 2D array
scaled_dataset_validation =  scaler.fit_transform(dataset_validation)  # scaling open values to between 0 and 1
print("Shape of scaled validation dataset :",scaled_dataset_validation.shape)

X_test = []
y_test = []

for i in range(time_step, length_validation):
    X_test.append(scaled_dataset_validation[i-time_step:i,0])
    y_test.append(scaled_dataset_validation[i,0])

X_test, y_test = np.array(X_test), np.array(y_test)
print("Shape of X_test before reshape :",X_test.shape)
print("Shape of y_test before reshape :",y_test.shape)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))  # reshape to 3D array
y_test = np.reshape(y_test, (-1,1))

print("Shape of X_test after reshape :",X_test.shape)
print("Shape of y_test after reshape :",y_test.shape)

# predictions with X_test data
y_pred_of_test = model_lstm.predict(X_test)
# scaling back from 0-1 to original
y_pred_of_test = scaler.inverse_transform(y_pred_of_test)
print("Shape of y_pred_of_test :",y_pred_of_test.shape)

# visualisation
plt.figure(figsize = (30,10))
plt.plot(y_pred_of_test, label = "y_pred_of_test", c = "orange")
plt.plot(scaler.inverse_transform(y_test), label = "y_test", c = "g")
plt.xlabel("Days")
plt.ylabel("Open price")
plt.title("LSTM model, Prediction with input X_test vs y_test")
plt.legend()
plt.show()

# Visualisation
plt.subplots(figsize =(30,12))
plt.plot(train_data.Date, train_data.Open, label = "train_data", color = "b")
plt.plot(validation_data.Date, validation_data.Open, label = "validation_data", color = "g")
plt.plot(train_data.Date.iloc[time_step:], y_pred, label = "y_pred", color = "r")
plt.plot(validation_data.Date.iloc[time_step:], y_pred_of_test, label = "y_pred_of_test", color = "orange")
plt.xlabel("Days")
plt.ylabel("Open price")
plt.title("LSTM model, Train-Validation-Prediction")
plt.legend()
plt.show()

data.iloc[-1]

X_input = data.iloc[-time_step:].Open.values               # getting last 50 rows and converting to array
X_input = scaler.fit_transform(X_input.reshape(-1,1))      # converting to 2D array and scaling
X_input = np.reshape(X_input, (1,50,1))                    # reshaping : converting to 3D array
print("Shape of X_input :", X_input.shape)
X_input

LSTM_prediction = scaler.inverse_transform(model_lstm.predict(X_input))

print("LSTM model, Open price prediction for 3/18/2017      :", LSTM_prediction[0,0])