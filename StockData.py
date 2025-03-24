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
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.features = []
        self.target = [] 

        if load_data == True:
            self.load_data()
            self.loaded = True
            print("Data")
            print(self.data.head(5))
            print(self.data.tail(5))
        else:
            self.loaded = False

        if create_model_data == True and self.loaded == True:
            self.split_data()
            self.model_data_created = True
            print("X_train")
            print(self.X_train.head(5))
            print(self.X_train.tail(5))
            print("X_test")
            print(self.X_test.head(5))
            print(self.X_test.tail(5))
            print("Y_train")
            print(self.Y_train.head(5))
            print(self.Y_train.tail(5))
            print("Y_test")
            print(self.Y_test.head(5))
            print(self.Y_test.tail(5))
        else:
            self.model_data_created = False


    def set_ticker(self, ticker, reload = False):
        """ Set new tickers and reload data if requested """
        self.ticker = ticker
        if reload == True:
            self.load_data()
            self.loaded = True
        else:
            self.loaded = False

    def set_dates(self, start_date, end_date, reload = False):
        """ Set new date range and  reload data if requested """
        self.start_date = start_date
        self.end_date = end_date
        if reload == True:
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

            print(df.head())
            print(df.tail())

            return True

        except Exception as e:

            print("Failure loading: ", self.ticker)

            print(e)

        return None


    def split_data(self):
        """Split the data into training and testing sets based on the train_date."""
        # self.data.dropna(inplace=True)
        # self.data = self.df.reset_index()  # Moves index columns back to normal columns
        # print(self.df.columns)

    # Ensure the index is a datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)

        # train_mask = self.data['date'] <= self.split_date
        # test_mask = self.data['date'] > self.split_date
        # train_data = self.data[train_mask]
        # test_data = self.data[test_mask]

        # Filter data using the index
        train_data = self.data[:self.split_date]
        test_data = self.data[self.split_date:]

        self.X_train = train_data[self.features]
        self.X_test = test_data[self.features]
        self.Y_train = train_data[self.target]
        self.Y_test = test_data[self.target]

        return

    def create_lagged_features(self, num_lags):
        """Create lagged features for the regression model."""
        if num_lags > 0:        
            for i in range(1, num_lags + 1):
                self.data['min_' + str(i) + '_close'] = self.data['close'].shift(i)
        elif num_lags < 0:
            self.data[f'fut_{-num_lags}_close'] = self.data['close'].shift(num_lags)

        self.data.dropna(inplace=True)  # Drop rows with NaN values created by shifting

        # Update train_data and test_data with the new features
        # train_mask = self.df['date'] <= self.train_date
        # test_mask = self.df['date'] > self.train_date
        # self.train_data = self.df[train_mask]
        # self.test_data = self.df[test_mask]


    def create_trend_features(self, num_lags):
        """Create lagged features for the regression model."""
        self.create_lagged_features(num_lags+1)

        overall_trend = 0
        for i in range(1, num_lags + 1):
            self.data['min_' + str(i) + '_trend'] = np.where(self.data['min_' + str(i) + '_close'] > self.data['min_' + str(i+1) + '_close'], 1, -1)
            overall_trend += self.data['min_' + str(i) + '_trend']

        self.data['trend_' + str(num_lags) + '_day'] = np.where(overall_trend > 0, 1, -1)

        # self.df['min_1_close'] = self.df['close'].shift(1)
        # self.df['min_2_close'] = self.df['close'].shift(2)
        self.data.dropna(inplace=True)  # Drop rows with NaN values created by shifting

        # Update train_data and test_data with the new features
        # train_mask = self.df['date'] <= self.train_date
        # test_mask = self.df['date'] > self.train_date
        # self.train_data = self.df[train_mask]
        # self.test_data = self.df[test_mask]


    # def prepare_features(self):
    # # def prepare_features(self, features, target):
    #     """Prepare features and target for training and testing."""
    #     self.df.dropna(inplace=True)
    #     self.X_train = self.train_data[self.features]
    #     self.X_test = self.test_data[self.features]
    #     self.Y_train = self.train_data[self.target]
    #     self.Y_test = self.test_data[self.target]
    #     return # X_train, X_test, y_train, y_test


    def standardise_input(self, feature):

        # mu = float(self.X_train[feature].mean())
        # sigma = float(self.X_train[feature].std())
        mu = float(self.X_train[feature].iloc[0])
        sigma = float(self.X_train[feature].iloc[0])

        stdize_input = lambda x: (x - mu) / sigma

        # X_train = X_train.apply(stdize_input)
        # X_test = X_test.apply(stdize_input)

        self.X_train = (self.X_train - mu) / sigma
        self.X_test = (self.X_test - mu) / sigma
       
        return