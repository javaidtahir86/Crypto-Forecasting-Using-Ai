#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as dtr
import yfinance as yf
from datetime import datetime
from pandas_datareader import data as pdr
from keras.models import load_model
import warnings
import streamlit as st
# # warnings.filterwarnings('ignore')


start_date=datetime(2015,12,31)
end_date=datetime(2023,5,30)

st.title("AI Based Crypto Prediction")
user_input=st.text_input("Enter your coin","BTC-USD")
yf.pdr_override()
data=pdr.get_data_yahoo(user_input,start=start_date,end=end_date)

#Describing Data

st.subheader('Data from 2016 - 2023')
st.write(data)

# #visualization
# st.subheader('Closing price ')
# fig=plt.figure(figsize=(12,6))
# plt.plot(data.Close)
# st.pyplot(fig)

group=data.groupby('Date')
gr_by_closing_rate=group['Close'].mean()

#spliting data into training and testing

prediction_days=60
df_train=gr_by_closing_rate[:len(data)-prediction_days].values.reshape(-1,1)
df_test=gr_by_closing_rate[len(data)-prediction_days:].values.reshape(-1,1)

#Scalling

from sklearn.preprocessing import MinMaxScaler

scaler_train=MinMaxScaler(feature_range=(0,1))# converting train data into 0s,and 1s
scaled_train=scaler_train.fit_transform(df_train)

scaler_test=MinMaxScaler(feature_range=(0,1))#converting test data into 0s,and 1s
scaled_test=scaler_test.fit_transform(df_test)

#DataSet Generator LSTM
def dataset_generator_lstm(dataset, look_back=5):
    dataX, dataY = [], []

    for i in range(len(dataset) - look_back):
        window_size_x = dataset[i:(i + look_back), 0]
        dataX.append(window_size_x)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


trainX, trainY = dataset_generator_lstm(scaled_train)
testX, testY = dataset_generator_lstm(scaled_test)

#Converting Data to 3D Tensor
trainX=np.reshape(trainX,(trainX.shape[0],trainX.shape[1],1))
testX=np.reshape(testX,(testX.shape[0],trainX.shape[1],1))

#Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import LSTM
#
# network=Sequential()
#
# network.add(LSTM(units=128, activation='relu',return_sequences=True, input_shape = (trainX.shape[1], trainX.shape[2])))
# network.add(Dropout(0.2))
#
# network.add(LSTM(units = 64, input_shape = (trainX.shape[1], trainX.shape[2])))
# network.add(Dropout(0.2))
#
# network.add(Dense(units=1))
# network.summary()
#
#
# #Compiling and Model Training
#
# from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
# network.compile(optimizer='adam',loss='mean_squared_error')
checkpoint_path='my_best_model.hdf5'
# checkpoint =  ModelCheckpoint(filepath=checkpoint_path,
#                           moniter='val_loss',
#                           save_best_only=True,
#                           mode='min')
#
# earlystopping  =  EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
# callbacks  =  [checkpoint,earlystopping]
#
# history  =  network.fit(trainX,trainY,batch_size=32,epochs=300,verbose=1,shuffle=False,
#             validation_data=(testX,testY),callbacks=callbacks)
#
#LSTM Predictions Using testX and plotting line Graph against Actual testY

model_from_saved_check_point=load_model(checkpoint_path)
predicted_coin_price_test_data = model_from_saved_check_point.predict(testX)
predicted_coin_price_test_data = scaler_test.inverse_transform(predicted_coin_price_test_data.reshape(-1,1))
test_actual = scaler_test.inverse_transform(testY.reshape(-1,1))
#Graph
#
# plt.figure(figsize=(16,9))
# plt.plot(predicted_coin_price_test_data,'r',marker='.' ,label='Predicted Test')
# plt.plot(test_actual,marker='.' ,label='Actual Test')
# plt.legend()
# plt.show()


st.subheader('Predicted Test vs Actual Test ')
fig=plt.figure(figsize=(12,6))
plt.plot(predicted_coin_price_test_data,'r',marker='.' ,label='Predicted Test')
plt.plot(test_actual,marker='.' ,label='Actual Test')
st.pyplot(fig)
#

# LSTM Prediction Using TrainX and plotting line graph against actual trainY
#
# predicted_coin_price_train_data = model_from_saved_check_point.predict(trainX)
#
# predicted_coin_price_train_data = scaler_test.inverse_transform(predicted_coin_price_train_data.reshape(-1,1))
#
# train_actual = scaler_test.inverse_transform(trainY.reshape(-1,1))
#
# #Graph
#
#
# st.subheader('Predicted Train vs Actual Train ')
# fig=plt.figure(figsize=(12,6))
# plt.plot(predicted_coin_price_train_data,'r',marker='.' ,label='Predicted Train')
# plt.plot(test_actual,marker='.' ,label='Actual Train')
# st.pyplot(fig)
#

#Predicting Next 5 Days Coin Price

lookback_period = 5
testX_last_5_days = testX[testX.shape[0] - lookback_period :  ]
predicted_5_days_forecast_price_test_x = []
for i in range(5):
    predicted_forecast_price_test_x = model_from_saved_check_point.predict(testX_last_5_days[i:i + 1])

    predicted_forecast_price_test_x = scaler_test.inverse_transform(predicted_forecast_price_test_x.reshape(-1, 1))
    # print(predicted_forecast_price_test_x)
    predicted_5_days_forecast_price_test_x.append(predicted_forecast_price_test_x)

predicted_5_days_forecast_price_test_x = np.array(predicted_5_days_forecast_price_test_x)

predicted_5_days_forecast_price_test_x = predicted_5_days_forecast_price_test_x.flatten()
predicted_coin_price_test_data = predicted_coin_price_test_data.flatten()

predicted_btc_test_concatenated = np.concatenate((predicted_coin_price_test_data, predicted_5_days_forecast_price_test_x))


#Final Graph


st.subheader('Predicted Next 5 Days  ')
fig=plt.figure(figsize=(12,6))
plt.plot(predicted_btc_test_concatenated,'b',marker='.' ,label='Prediction')
plt.plot(test_actual,marker='.' ,label='Actual Test')
st.pyplot(fig)
