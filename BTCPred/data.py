import pandas as pd
import numpy as np
def get_data_csv():
    data = pd.read_csv(
        '../raw_data/FTX_BTCUSD_1h.csv', header=1)
    print('data retrieved')
    return data


def clean_data(data):
    data = data.iloc[::-1]
    return data

def sequencing(data):
    SEQUENCE_SIZE = 11
    values = data['Close']
    res = np.zeros(shape=(values.shape[0] - SEQUENCE_SIZE, SEQUENCE_SIZE))
    for i in range(values.shape[0] - SEQUENCE_SIZE):
        seq = values[i:i+SEQUENCE_SIZE]
        res[i,:] = seq
    X = res[:,:-1]
    y = res[:,-1]
    X_test = res[-1, 1:]
    return X, y, X_test

if __name__ == '__main__':
    #data = get_data_csv()
    #data = clean_data()
    #data = sequencing()
    pass
