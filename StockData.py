import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

class StockData:
    def __init__(self, ticker, start_date='2014-12-31', end_date='2024-12-31'):
        """ Initialise StockData object by loading data"""
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.load_data()

    def set_ticker_and_dates(self, tickers, start_date, end_date):
        """ Set new tickers and date range, then reload data """
        self.ticker = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.load_data()

    def add_lagged_price(self, lags, *args, **kwargs):
        """ Method to add lagged prices to the data. """
        if self.data is not None:
            # Compute n-day lag close price within each ticker group
            for lag in lags:
                self.data['lag_' + str(lag) + '_close'] = self.data['close'].shift(lag)
        else:
            print("No data loaded. Cannot add indicator.")

    def add_indicator(self, indicator_func, *args, **kwargs):
        """ Method to add new indicators to the data. """
        if self.data is not None:
            self.data = indicator_func(self.data, *args, **kwargs)
        else:
            print("No data loaded. Cannot add indicator.")
        
    def get_data(self):
        """ Return the dataframe with all indicators added. """
        return self.data

    # def print_stats_tests(self):
    #     """ Prints results of Augmented Dickey-Fuller test. """
    #     adf, p, usedlag, nobs, cvs, aic = sm.tsa.stattools.adfuller(self.data['close'])
    #     adf_results_string = 'ADF: {}\np-value: {},\nN: {}, \ncritical values: {}'
    #     print(adf_results_string.format(adf, p, nobs, cvs))

    #     pacf = sm.tsa.stattools.pacf(self.data['close'], nlags=30)
    #     T = len(self.data['close'])

    #     sig_test = lambda tau_h: np.abs(tau_h) > 2.58/np.sqrt(T)

    #     for i in range(len(pacf)):
    #         if sig_test(pacf[i]) == False:
    #             n_steps = i - 1
    #             print('n_steps set to', n_steps)
    #             break

    #     plt.plot(pacf, label='pacf')
    #     plt.plot([2.58/np.sqrt(T)]*30, label='99% confidence interval (upper)')
    #     plt.plot([-2.58/np.sqrt(T)]*30, label='99% confidence interval (lower)')
    #     plt.xlabel('number of lags')
    #     plt.xticks(np.arange(0, 30, 2))
    #     plt.legend()
    #     plt.show()

    #     return 

    def get_stats_tests(self):
        """ Returns results of Augmented Dickey-Fuller test as a string. """
        adf, p, usedlag, nobs, cvs, aic = sm.tsa.stattools.adfuller(self.data['close'])
        adf_results_string = 'ADF: {}\np-value: {},\nN: {}, \ncritical values: {}'.format(adf, p, nobs, cvs)

        pacf = sm.tsa.stattools.pacf(self.data['close'], nlags=30)
        T = len(self.data['close'])

        sig_test = lambda tau_h: np.abs(tau_h) > 2.58/np.sqrt(T)

        n_steps = None
        for i in range(len(pacf)):
            if sig_test(pacf[i]) == False:
                n_steps = i - 1
                break

        pacf_results_string = 'PACF n_steps: {}'.format(n_steps)
        return adf_results_string + '\n' + pacf_results_string

    def load_data(self):

        try:
            df = yf.download(tickers=self.ticker,
                        start=self.start_date,
                        end=self.end_date)


            df.index.name = 'date'

            # Flatten MultiIndex columns if necessary
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = df.columns.str.lower()

            self.data = df

            print(self.ticker, " loaded.")

            return True

        except Exception as e:

            print("Failure loading: ", self.ticker)

            print(e)

        return None