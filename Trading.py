import pandas as pd
import numpy as np
import yfinance as yf

# from LoadData import load_data
from StockData import StockData

# Create StockData object
# stock_data = StockData(data)

# asx200 = pd.read_html('https://en.wikipedia.org/wiki/S%26P/ASX_200')[2]

# asx200['Ticker'] = asx200['Code'].str.replace('.', '-').astype(str) + '.ax'
# asx200['Symbol'] = asx200['Code'].str.replace('.', '-')

# tickers = asx200['Ticker'].unique().tolist()

ticker = "WBC.AX"

end_date = '2024-12-31'

start_date = pd.to_datetime(end_date)-pd.DateOffset(365*10)

stock = StockData(ticker, start_date, end_date)

lags = [1, 2, 4]

stock.add_lagged_price(lags)

df = stock.get_data()

assert isinstance(df, pd.DataFrame), "Data is not a Pandas DataFrame"

print(df.head())

stock.print_stats_tests()

train_weight = 0.8
split = int(len(df)*train_weight)

df_train = df['close'].iloc[:split]
df_test = df['close'].iloc[split:]

mu = float(df_train.mean())
sigma = float(df_train.std())

stdize_input = lambda x: (x - mu) / sigma

df_train = df_train.apply(stdize_input)
df_test = df_test.apply(stdize_input)