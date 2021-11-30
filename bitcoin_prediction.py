from enum import auto
import streamlit as st
import datetime
import pandas as pd
import numpy as np
import json
import websocket
import binance
from PIL import Image


st.title("Cryptocurrencies Prediction App")

with st.expander("Useful links"):
    url1 = 'https://www.coindesk.com/'
    st.markdown("[CoinDesk: Crypto news and related affairs](%s)" % url1)
    url2 = 'https://www.binance.com/en'
    st.markdown("[Binance: Crypto Exchange plateform](%s)" % url2)
    url3 = 'https://coinmarketcap.com/'
    st.markdown("[CoinMarketCap: Crypto data agreggator](%s)" % url3)


client = binance.Client()

cc = st.selectbox('What crypto do you want to trade?', ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'], help= 'BTC, ETH and BNB paired with USD tether (stablecoin pegged to $USD)')

cc2 = cc.lower()

interval = st.selectbox('What interval do you want to trade in?',
                        ['1m', '1w', '1d', '1h', '5m', '1M'],
                        help='m=minute, h=hour, d=day, w=week, M=month')

socket = f'wss://stream.binance.com:9443/ws/{cc2}@kline_{interval}'

start_date = st.date_input("Start Date?", datetime.date(2021, 11, 26))
start_str = str(start_date.day) + ' ' + str(datetime.date(1900, start_date.month, 1).strftime('%b')) + ', ' + str(start_date.year)

end_date = st.date_input("End Date?", datetime.date(2021, 11, 27))
end_str = str(end_date.day) + ' ' + str(datetime.date(1900, end_date.month, 1).strftime('%b')) + ', ' + str(end_date.year)

result = client.get_historical_klines(cc, interval, start_str=start_str, end_str=end_str)
if st.button('Make me a prediction cryptobro'):
    
    st.write('here is you prediction babe')


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
        st.write(f"""Closes: {cs['c']}""")
        st.write(df)


st.sidebar.header('Available Tokens 💰 :')
image = Image.open('/Users/noam/code/nonoperry/data-challenges/BTClogo.png')
st.sidebar.image(image, width=75, caption='Bitcoin')
image = Image.open('/Users/noam/code/nonoperry/data-challenges/ETHlogo.png')
st.sidebar.image(image, width=75, caption='Ethereum')
image = Image.open('/Users/noam/code/nonoperry/data-challenges/BNBlogo.png')
st.sidebar.image(image, width=75, caption='Binance Coin')
st.sidebar.header('Sources 📖 :')
url = 'https://www.binance.com/fr/support/faq/360002502072'
st.sidebar.markdown("check out this [link](%s)" % url)


ws = websocket.WebSocketApp(socket, on_message=on_message, on_close=on_close)

ws.run_forever()
