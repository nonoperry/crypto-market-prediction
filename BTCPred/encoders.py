from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


def scaler(X, y):
    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1,1))
    X_scaled = X_scaled.reshape(-1,1,10)
    y_scaled = y_scaled.reshape(-1,1)
    return X_scaled, y_scaled, scaler_x, scaler_y

def split(X_scaled,y_scaled):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2)
    return X_train, X_test, y_train, y_test
