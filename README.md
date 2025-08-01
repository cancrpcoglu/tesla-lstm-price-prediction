# Tesla Stock Price Prediction using LSTM

## 📌 Project Overview

This project focuses on predicting Tesla Inc. stock prices using historical data and Long Short-Term Memory (LSTM) networks. The model utilizes a recurrent neural network (RNN)-based architecture to capture temporal dependencies in the time series data and forecast future closing prices.

## 🧰 Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- Scikit-Learn
- Dataset from Kaggle (Tesla Stock CSV)

## 🗂️ Dataset

- Source: [Kaggle - Tesla Stock Dataset]([https://www.kaggle.com/](https://www.kaggle.com/code/serkanp/tesla-stock-price-prediction)
- Features used: `Open`, `High`, `Low`, `Close`, `Volume`
- 70% of the dataset was used for training, 30% for testing

## 🧠 Model Summary

- LSTM layers were used to build a time-series forecasting model
- Several hyperparameters were optimized (e.g., number of epochs, dense layers, optimizers)
- Techniques such as `EarlyStopping` were applied to reduce overfitting
- Evaluation Metrics: Mean Squared Error, Accuracy, Loss

## 📊 Results

- Achieved **87% accuracy** on the test dataset (replace with your actual result)
- Visualized performance metrics using `matplotlib`


## 🚀 How to Run

pip install -r requirements.txt
python lstm_model.py

## ✍️ Author
Can Çorapçıoğlu
[GitHub](https://github.com/cancrpcoglu) | [LinkedIn](https://www.linkedin.com/in/can-%C3%A7orap%C3%A7%C4%B1o%C4%9Flu-15a340247/)
