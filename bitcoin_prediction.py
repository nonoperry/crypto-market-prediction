import streamlit as st
import datetime
import pandas as pd
import numpy as np
import json
import websocket
from BTCPred.trainer import Model
from BTCPred.data import get_data_csv, clean_data, sequencing
from BTCPred.encoders import scaler, split
from sklearn.preprocessing import MinMaxScaler
import binance

st.markdown("""# Cryptocurrencies Prediction App""")

client = binance.Client()

cc = st.selectbox('What crypto do you want to trade?', ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])

cc2 = cc.lower()

interval = st.selectbox('What interval do you want to trade in?',
                        ['1m', '5m', '1h', '1d', '1w', '1M'])

socket = f'wss://stream.binance.com:9443/ws/{cc2}@kline_{interval}'

start_date = st.date_input("Start Date?", datetime.date(2021, 11, 30))
start_str = str(start_date.day) + ' ' + str(datetime.date(1900, start_date.month, 1).strftime('%b')) + ', ' + str(start_date.year)

end_date = st.date_input("End Date?", datetime.date(2021, 12, 1))
end_str = str(end_date.day) + ' ' + str(datetime.date(1900, end_date.month, 1).strftime('%b')) + ', ' + str(end_date.year)

result = client.get_historical_klines(cc, interval, start_str=start_str, end_str=end_str)

df = pd.DataFrame(result)

df[0] = pd.to_datetime(df[0], unit='ms')
df[6] = pd.to_datetime(df[6], unit='ms')

df.rename(columns={0: 'Open Time',
                   1: 'Open',
                   2: 'High',
                   3: 'Low',
                   4: 'Close',
                   5: 'Volume',
                   6: 'Close Time',
                   7: 'Quote Asset Volume',
                   8: 'Number of Trades',
                   9: 'Taker Buy Base Asset Volum',
                   10: 'Taker Buy Quote Asset Volume',
                   11: 'Ignore'}, inplace=True)

df.drop(df.tail(1).index,inplace=True)
df

X, y, X_test = sequencing(df)

X_scaled, y_scaled, scaler_x, scaler_y = scaler(X, y)

X_train, X_test, y_train, y_test = split(X_scaled, y_scaled)

test = Model(X_train, y_train, scaler_y, scaler_x, X_scaled)

model = test.lstm_model()

es = test.early_stop()

test.fit()

st.write(test.graphing(y))

def on_close(ws):
    st.markdown("""All trades settled.""")

def on_message(ws, message):
    global df
    json_message = json.loads(message)
    cs = json_message['k']
    if cs['x']:
        new_row = {'Open Time': cs['t'],
                   'Open': cs['o'],
                   'High': cs['h'],
                   'Low' : cs['l'],
                   'Close': cs['c'],
                   'Volume': cs['v'],
                   'Close Time': cs['T'],
                   'Quote Asset Volume': cs['q'],
                   'Number of Trades': cs['n'],
                   'Taker Buy Base Asset Volum': cs['V'],
                   'Taker Buy Quote Asset Volume': cs['Q'],
                   'Ignore': cs['B']}
        df = df.append(new_row, ignore_index=True)
        df.loc[len(df) - 1, 'Open Time'] = pd.to_datetime(df.loc[len(df) - 1, 'Open Time'], unit='ms')
        df.loc[len(df) - 1, 'Close Time'] = pd.to_datetime(df.loc[len(df) - 1, 'Close Time'], unit='ms')
        df['Close'] = df['Close'].astype(float)
        st.write(f"""Closes: {cs['c']}""")
        st.write(df)

        X_test = X_test = df['Close'].values[-10:]
        X_test = X_test.reshape(1, -1)
        X_test_scaled = scaler_x.transform(X_test)
        X_test_scaled = X_test_scaled.reshape(-1, 1, 10)
        st.write("Last prices :", X_test)
        prediction = test.model.predict(X_test_scaled.reshape((1, 1, 10)))
        prediction = prediction.reshape((1, 1))
        st.write("Prediction :", test.scaler_y.inverse_transform(prediction))

ws = websocket.WebSocketApp(socket, on_message=on_message, on_close=on_close)

ws.run_forever()
