import streamlit as st
import datetime
import pandas as pd
import json
import websocket
from BTCPred.trainer import Model
from BTCPred.data import sequencing
from BTCPred.encoders import scaler, split
import binance
from PIL import Image
import altair as alt


st.title("Cryptocurrencies Prediction App")

with st.expander("Useful links"):
    url1 = 'https://www.coindesk.com/'
    st.markdown("[CoinDesk: Crypto news and related affairs](%s)" % url1)
    url2 = 'https://www.binance.com/en'
    st.markdown("[Binance: Crypto Exchange plateform](%s)" % url2)
    url3 = 'https://coinmarketcap.com/'
    st.markdown("[CoinMarketCap: Crypto data agreggator](%s)" % url3)

st.sidebar.header('Available Tokens 💰 :')
image = Image.open('raw_data/BTClogo.png')
st.sidebar.image(image, width=75, caption='Bitcoin')
image = Image.open('raw_data/ETHlogo.png')
st.sidebar.image(image, width=75, caption='Ethereum')
image = Image.open('raw_data/BNBlogo.png')
st.sidebar.image(image, width=75, caption='Binance Coin')
st.sidebar.header('Sources 📖 :')
url = 'https://www.binance.com/fr/support/faq/360002502072'
st.sidebar.markdown("check out this [link](%s)" % url)

client = binance.Client()

cc = st.selectbox('What crypto do you want to trade?', ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'], help= 'BTC, ETH and BNB paired with USD tether (stablecoin pegged to $USD)')

cc2 = cc.lower()

interval = st.selectbox('What interval do you want to trade in?',
                        ['1m', '5m', '1h', '1d', '1w', '1M'],
                        help='m=minute, h=hour, d=day, w=week, M=month')

socket = f'wss://stream.binance.com:9443/ws/{cc2}@kline_{interval}'

start_date = st.date_input("Start Date?", datetime.date(2021, 12, 1))
start_str = str(start_date.day) + ' ' + str(datetime.date(1900, start_date.month, 1).strftime('%b')) + ', ' + str(start_date.year)

end_date = st.date_input("End Date?", datetime.date(2021, 12, 2))
end_str = str(end_date.day) + ' ' + str(datetime.date(1900, end_date.month, 1).strftime('%b')) + ', ' + str(end_date.year)

result = client.get_historical_klines(cc, interval, start_str=start_str, end_str=end_str)

df = pd.DataFrame(result)

df = df.iloc[:, :7]

prediction_time = pd.to_datetime(
    df[0].iloc[-1],
    unit='ms').tz_localize('UTC').tz_convert('Europe/Brussels').strftime('%Y-%m-%d %H:%M:%S')

df[0] = pd.to_datetime(df[0], unit='ms').dt.tz_localize('UTC').dt.tz_convert(
    'Europe/Brussels').dt.strftime('%Y-%m-%d %H:%M:%S')
df[1] = df[1].astype(float)
df[2] = df[2].astype(float)
df[3] = df[3].astype(float)
df[4] = df[4].astype(float)
df[5] = df[5].astype(float)
df[7] = pd.to_datetime(df[6], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/Brussels')
df[6] = pd.to_datetime(df[6], unit='ms').dt.tz_localize('UTC').dt.tz_convert(
    'Europe/Brussels').dt.strftime('%Y-%m-%d %H:%M:%S')

df.rename(columns={
    0: 'Open Time',
    1: 'Open',
    2: 'High',
    3: 'Low',
    4: 'Close',
    5: 'Volume',
    6: 'Close Time',
    7: 'Cls Time'
}, inplace=True)

df.drop(df.tail(1).index,inplace=True)

X, y, X_test = sequencing(df)

X_scaled, y_scaled, scaler_x, scaler_y = scaler(X, y)

X_train, X_test, y_train, y_test = split(X_scaled, y_scaled)

test = Model(X_train, y_train, scaler_y, scaler_x, X_scaled)

model = test.lstm_model()

es = test.early_stop()

test.fit()

test.graphing(y)

def on_close(ws):
    st.markdown("Done")

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
                   'Cls Time': cs['T']}
        df = df.append(new_row, ignore_index=True)

        prediction_time = (pd.to_datetime(
            df['Open Time'].iloc[-1], unit='ms').tz_localize('UTC').tz_convert(
                'Europe/Brussels') + pd.Timedelta(
                    minutes=2)).strftime('%Y-%m-%d %H:%M:%S')
        df['Open Time'].iloc[-1] = pd.to_datetime(
            df['Open Time'].iloc[-1], unit='ms').tz_localize('UTC').tz_convert(
                'Europe/Brussels').strftime('%Y-%m-%d %H:%M:%S')
        df['Open'].iloc[-1] = float(df['Open'].iloc[-1])
        df['High'].iloc[-1] = float(df['High'].iloc[-1])
        df['Low'].iloc[-1] = float(df['Low'].iloc[-1])
        df['Close'].iloc[-1] = float(df['Close'].iloc[-1])
        df['Volume'].iloc[-1] = float(df['Volume'].iloc[-1])
        df['Close Time'].iloc[-1] = pd.to_datetime(
            df['Close Time'].iloc[-1],
            unit='ms').tz_localize('UTC').tz_convert(
                'Europe/Brussels').strftime('%Y-%m-%d %H:%M:%S')
        df['Cls Time'].iloc[-1] = pd.to_datetime(
            df['Cls Time'].iloc[-1],
            unit='ms').tz_localize('UTC').tz_convert('Europe/Brussels')

        X_test = df['Close'].values[-10:]
        X_test = X_test.reshape(1, -1)
        X_test_scaled = scaler_x.transform(X_test)
        X_test_scaled = X_test_scaled.reshape(-1, 1, 10)
        st.write(f"Last price '{df['Close Time'].iloc[-1]}' :", X_test[0][9])
        prediction = test.model.predict(X_test_scaled.reshape((1, 1, 10)))
        prediction = prediction.reshape((1, 1))
        st.write(f"Prediction for '{prediction_time}' :", test.scaler_y.inverse_transform(prediction)[0][0])


X_test = df['Close'].values[-10:]
X_test = X_test.reshape(1, -1)
X_test_scaled = scaler_x.transform(X_test)
X_test_scaled = X_test_scaled.reshape(-1, 1, 10)
st.write(f"Last price '{df['Close Time'].iloc[-1]}' :", X_test[0][9])
prediction = test.model.predict(X_test_scaled.reshape((1, 1, 10)))
prediction = prediction.reshape((1, 1))
st.write(
    f"Prediction for '{prediction_time} + {interval}' :",
    test.scaler_y.inverse_transform(prediction)[0][0])

df_graph = pd.DataFrame({
    "price": [float(t) for t in df['Close']],
    "date": [t.to_pydatetime() for t in df['Cls Time']]})
df_graph = df_graph[-30:]
my_chart = alt.Chart(df_graph).mark_line(color="#FFAA00").encode(alt.Y('price', scale=alt.Scale(domain=(df_graph['price'].min(),
                                                                                         df_graph['price'].max()))),
                                                  alt.X('date:T')).properties(height=500, width=700)
st.altair_chart(my_chart)

ws = websocket.WebSocketApp(socket, on_message=on_message, on_close=on_close)

if interval == '1m':
    ws.run_forever()
