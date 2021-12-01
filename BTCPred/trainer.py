import keras
import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import optimizers
from BTCPred.data import get_data_csv, clean_data, sequencing
from BTCPred.encoders import scaler, split
import pandas as pd
import seaborn as sns
import streamlit as st


class Model(object):
    def __init__(self, X_train, y_train, scaler_y, scaler_x, X_scaled):
        self.X_train = X_train
        self.y_train = y_train
        self.scaler_y = scaler_y
        self.scaler_x = scaler_x
        self.X_scaled = X_scaled

    def lstm_model(self):
        model = Sequential()
        model.add(LSTM(units = 128, activation='relu', return_sequences = True, input_shape=(None,10)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=256, activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='mean_squared_error',metrics=['mse'])
        self.model = model

    def early_stop(self):
        es = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
        self.es = es

    def fit(self):
        history = self.model.fit(self.X_train, self.y_train, validation_split=0.1, batch_size=32 , epochs=100, shuffle=True,callbacks=self.es)
        self.history = history

    def graphing(self, y):
        original = pd.DataFrame(y)
        predictions = pd.DataFrame(self.scaler_y.inverse_transform(self.model.predict(self.X_scaled).reshape(-1,1)))
        sns.set(rc={'figure.figsize':(11.7+2,8.27+2)})
        ax = sns.lineplot(x=original.index, y = original[0], label='Test Data', color='royalblue')
        ax = sns.lineplot(x=original.index, y=predictions[0], label='Prediction', color='tomato')
        ax.set_title('BTC price',size=14, fontweight='bold')
        ax.set_xlabel('hours', size=14)
        ax.set_ylabel('Price(USD)',size=14)
        ax.set_xticklabels('',size=10)

    def predict(self, X_predict):
        prediction = self.model.predict(X_predict.reshape((1, 1, 10)))
        prediction = prediction.reshape((1, 1))
        self.scaler_y.inverse_transform(prediction)
        return prediction
