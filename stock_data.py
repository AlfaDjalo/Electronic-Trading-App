""" Data object for time series for a stock """

import pandas as pd
import numpy as np
import yfinance as yf
# import statsmodels.api as sm
# import matplotlib.pyplot as plt

class StockData:
    def __init__(self, ticker, start_date='2014-12-31', end_date='2024-12-31', split_date='2022-12-31', load_data=False, create_model_data=False):
        """ Initialise StockData object """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

        self.split_date = split_date
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.feature_column = 'close'
        self.features = []
        self.target = []

        if load_data is True:
            self.load_data()
            self.loaded = True
            # print("Data")
            # print(self.data.head(5))
            # print(self.data.tail(5))
        else:
            self.loaded = False

        if create_model_data is True and self.loaded is True:
            self.split_data()
            self.model_data_created = True
        else:
            self.model_data_created = False


    def set_ticker(self, ticker, reload = False):
        """ Set new tickers and reload data if requested """
        self.ticker = ticker
        if reload is True:
            self.load_data()
            self.loaded = True
        else:
            self.loaded = False

    def set_dates(self, start_date, end_date, reload = False):
        """ Set new date range and  reload data if requested """
        self.start_date = start_date
        self.end_date = end_date
        if reload is True:
            self.load_data()
            self.loaded = True
        else:
            self.loaded = False

    def get_ticker(self):
        """ Return the dataframe with all indicators added. """
        return self.ticker

    def get_dates(self):
        """ Return the dataframe with all indicators added. """
        return { "start_date":self.start_date, "end_date": self.end_date }

    def get_data(self):
        """ Return the dataframe with all indicators added. """
        return self.data

    def get_data2(self):
        return {'x_train': self.x_train, 'y_train': self.y_train, 'x_test': self.x_test, 'y_test': self.y_test }

    def load_data(self):
        """ Load data for ticker between start_date and end_date. """
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

            nof_missing_values = sum(np.isnan(df['close']))

            if nof_missing_values > 0:
                print(nof_missing_values, 'observations are missing.')
                print('This is {:.3f}% of the total.'.format(nof_missing_values*100/len(df)))


                df['close'] = df['close'].bfill()
                nof_missing_values = sum(np.isnan(df['close']))
                print('Now', nof_missing_values, 'observations are missing.')

            # print(df.head())
            # print(df.tail())

            return True

        except Exception as e:

            print("Failure loading: ", self.ticker)

            print(e)

        return None


    def split_data(self):
        """Split the data into training and testing sets based on the train_date."""
        # Ensure the index and split_date are datetime objects
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        self.split_date = pd.to_datetime(self.split_date)

        # Split the data
        train_data = self.data[:self.split_date]
        test_data = self.data[self.split_date:]

        self.x_train = train_data[self.features]
        self.x_test = test_data[self.features]
        self.y_train = train_data[self.target]
        self.y_test = test_data[self.target]
        return

    def create_lagged_features(self, num_lags):
        """Create lagged features for the regression model."""
        if num_lags > 0:
            for i in range(1, num_lags + 1):
                self.data[f'min_{i}_{self.feature_column}'] = self.data[self.feature_column].shift(i)
        elif num_lags < 0:
            self.data[f'fut_{-num_lags}_{self.feature_column}'] = self.data[self.feature_column].shift(num_lags)

        self.data.dropna(inplace=True)  # Drop rows with NaN values created by shifting

    def create_trend_features(self, num_lags):
        """Create lagged trend features for the regression model."""
        self.create_lagged_features(num_lags + 1)

        overall_trend = 0
        for i in range(1, num_lags + 1):
            self.data[f'min_{i}_trend'] = np.where(
                self.data[f'min_{i}_{self.feature_column}'] > self.data[f'min_{i+1}_{self.feature_column}'], 1, -1
            )
            overall_trend += self.data[f'min_{i}_trend']

        self.data[f'trend_{num_lags}_day'] = np.where(overall_trend > 0, 1, -1)
        self.data.dropna(inplace=True)  # Drop rows with NaN values created by shifting

    def standardise_input(self, features):
        """Standardise the input features."""
        for feature in features:
            mu = float(self.x_train[feature].mean())
            sigma = float(self.x_train[feature].std())
            self.x_train[feature] = (self.x_train[feature] - mu) / sigma
            self.x_test[feature] = (self.x_test[feature] - mu) / sigma

    def calculate_log_returns(self):
        """Calculate log returns for the close prices."""
        if 'close' in self.data.columns:
            self.data['log_returns'] = np.log(self.data['close'] / self.data['close'].shift(1))
            self.data.dropna(inplace=True)  # Remove rows with NaN values caused by shifting
            self.feature_column = 'log_returns'  # Update feature column to log returns
        else:
            raise ValueError("Close prices are not available in the data.")

    def create_feature_list(self, comparison):
        """Generate the list of required features based on the comparison details."""
        model = comparison['model']
        parameters = comparison['params']

        # Handle log returns if enabled in the comparison
        if comparison.get('use_log_returns', False):
            self.calculate_log_returns()

        # Handle lagged features
        num_days_lag = int(parameters.get('num_days_lag', {}).get('value', 0))
        if num_days_lag > 0:
            self.create_lagged_features(num_days_lag)

        # Handle trend features for models that require them
        if model == 'LinearRegression' and 'trend_features' in parameters:
            trend_days = int(parameters.get('trend_features', {}).get('value', 0))
            if trend_days > 0:
                self.create_trend_features(trend_days)

        self.features = [col for col in self.data.columns if col.startswith('min_') or col.startswith('trend_')]
        return self.features

    def create_target(self, comparison):
        """Generate the target based on the comparison details."""
        parameters = comparison['params']
        forward_projection_days = int(parameters.get('forward_projection_days', {}).get('value', 0))

        if forward_projection_days > 1:
            self.create_lagged_features(-forward_projection_days)

        if forward_projection_days == 1:
            self.target = [self.feature_column]
        else:
            self.target = [f'fut_{forward_projection_days}_{self.feature_column}']

        return self.target

