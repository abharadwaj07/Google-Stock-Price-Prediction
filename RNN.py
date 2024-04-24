# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn 
import tensorflow

# Importing Train dataset
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

# Part 1 - Data Pre-Processing
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #1 in the end because we only have 1 indicator or feature ("Open" - in this case)

# Part 2 - Building RNN

# Importing Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# Initializing RNN
regressor = Sequential()

# Adding first LSTM layer and some dropout regularisation to avoid overfitting
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding second LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding third LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


# Adding fourth LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding Output layer
regressor.add(Dense(units = 1))

# Compiling RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Training RNN
regressor.fit(X_train, y_train, batch_size =32 , epochs = 100)

# Part 3 - Making predictions and visualizing the results

# Getting the real stock price of Google of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of Google of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the results
plt.plot(real_stock_price, color = 'Red', label = "Real Google Stock Price")
plt.plot(predicted_stock_price, color = 'Blue', label = "Predicted Google Stock Price")
plt.title("Google Stock Price Prediction for January 2017")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()















