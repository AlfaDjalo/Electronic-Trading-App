import pandas as pd
import yfinance as yf
# import numpy as np

def load_data_multiple(tickers, start_date='2018-12-31', end_date='2020-12-31'):

    df = yf.download(tickers=tickers,
                    start=start_date,
                    end=end_date).stack()

    df.index.names = ['date', 'ticker']

    df.columns = df.columns.str.lower()

    df = df.drop(columns=['adj close'])


def load_data_single(ticker, start_date='2018-12-31', end_date='2020-12-31'):

    df = yf.download(tickers=ticker,
                    start=start_date,
                    end=end_date).stack()

    df.index.names = ['date']

    df.columns = df.columns.str.lower()

    df = df.drop(columns=['adj close'])
