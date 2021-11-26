import keras
import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import optimizers




class Model(object):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def lstm_model(self):

        model = Sequential()
        model.add(LSTM(units = 128, activation='relu', return_sequences = True, input_shape=(None,10)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=256, activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='linear'))
        model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='mean_squared_error',metrics=['mse'])
        return model

    def early_stop(self):
        es = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
        return es

    def fit(model,es, X_train, y_train):
        history = model.fit(X_train, y_train, validation_split=0.1, batch_size=32 , epochs=100,shuffle=True,callbacks=es)
        return history

    print('salut')
