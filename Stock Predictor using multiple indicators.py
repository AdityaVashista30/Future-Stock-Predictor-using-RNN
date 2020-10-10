
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:52:49 2020

@author: Hp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#STEP 1:DATA PREPROCESSING on training set
dataset_train2=pd.read_csv("Google_Stock_Price_Train.csv")
training_set2=dataset_train2.iloc[:,1:5].values

#FEATURE SCALING
from sklearn.preprocessing import MinMaxScaler
sc2=MinMaxScaler(feature_range=(0,1))
training_set_SC2=sc2.fit_transform(training_set2)

# Creating a data structure with 60 timesteps and 1 output
"""RNN WILL SEE 60 past info, then predict next output"""
x_train2=[]
y_train2=[]

for i in range(60,len(training_set_SC2)):
    x_train2.append(training_set_SC2[i-60:i,:])
    y_train2.append(training_set_SC2[i,0])
    
x_train2,y_train2=np.array(x_train2),np.array(y_train2)

#RESHAPE
x_train2=np.reshape(x_train2,(x_train2.shape[0],x_train2.shape[1],4))


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# Initialising the RNN
regressor2 = Sequential()

#ADDING 1st LSTM layer
regressor2.add(LSTM(units=50,return_sequences = True,input_shape=(x_train2.shape[1],4)))
#(no, of timestem, no of indicatores)
regressor2.add(Dropout(rate=0.2))

#2nd Layer
regressor2.add(LSTM(units = 50, return_sequences = True))
regressor2.add(Dropout(0.2))

# Adding a 3rd Layer
regressor2.add(LSTM(units = 50, return_sequences = True))
regressor2.add(Dropout(0.2))

# Adding a 4th Layer
regressor2.add(LSTM(units = 50))#return_sequences = False
regressor2.add(Dropout(0.2))

#Adding Output LAyer
regressor2.add(Dense(units=1))

#COMPILING
regressor2.compile(optimizer='adam',loss='mean_squared_error')
#FITTING
regressor2.fit(x_train2,y_train2,batch_size=32,epochs=150)

#PREDICTING
dataset_test2=pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price2=dataset_test2.iloc[:,1].values

data_set_total2=pd.concat((dataset_train2[['Open','High','Low'],'Close'],dataset_test2[['Open','High','Low','Close']]),axis=0)#vertical concatination; adding lines

inputs2=data_set_total2[len(data_set_total2)-len(dataset_test2)-60:].values
inputs2=sc2.transform(inputs2)

x_test2=[]
for i in range(60,60+len(dataset_test2)):
    x_test2.append(inputs2[i-60:i,:])
        
x_test2=np.array(x_test2)
x_test2=np.reshape(x_test2,(x_test2.shape[0],x_test2.shape[1],4))


stock_price_predicted2=regressor2.predict(x_test2)
"""WE NEED TRANSFORM THE RESULTS TO ARRAY OF 3 columns in order to apply reverse transform
then we need to single out the results into array of 1 column"""
stock_price_predicted2=np.append(stock_price_predicted2,np.ones(len(stock_price_predicted2)).reshape(-1,1),axis=1)
stock_price_predicted2=np.append(stock_price_predicted2,np.ones(len(stock_price_predicted2)).reshape(-1,1),axis=1)
stock_price_predicted2=sc2.inverse_transform(stock_price_predicted2)
stock_price_predicted2=stock_price_predicted2[:,0]
stock_price_predicted2=stock_price_predicted2.reshape(-1,1)


plt.plot(real_stock_price2, color = 'red', label = 'Real Google Stock Price')
plt.plot(stock_price_predicted2, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

regressor2.save("StockPredictor.h5")
