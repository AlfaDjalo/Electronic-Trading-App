""" Data object for time series for a stock """

import pandas as pd
import numpy as np
import yfinance as yf
# import statsmodels.api as sm
# import matplotlib.pyplot as plt

# Global constants for default dates
DEFAULT_START_DATE = '2014-12-31'
DEFAULT_END_DATE = '2024-12-31'

class StockData:
    def __init__(self, ticker, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE, split_date='2022-12-31', load_data=False, create_model_data=False, use_lob_data=False, lob_filepath=None):
        """
        Initialise StockData object.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str): Start date for the data (format: 'YYYY-MM-DD').
            end_date (str): End date for the data (format: 'YYYY-MM-DD').
            split_date (str): Date to split data into training and testing sets.
            load_data (bool): Whether to load stock data immediately.
            create_model_data (bool): Whether to create model data immediately.
            use_lob_data (bool): Whether to use limit order book (LOB) data.
            lob_filepath (str): Filepath for LOB data CSV file.
        """
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

        self.use_lob_data = use_lob_data
        self.lob_filepath = lob_filepath
        self.lob_data = None  # Initialize LOB data attribute

        if load_data is True:
            self.load_data()
            self.loaded = True
        else:
            self.loaded = False

        if self.use_lob_data and self.lob_filepath:
            self.load_lob_data(self.lob_filepath)

        if create_model_data is True and self.loaded is True:
            self.split_data() # !!! Need to use full
            self.model_data_created = True
        else:
            self.model_data_created = False

    def set_ticker(self, ticker, reload = False):
        """
        Set a new ticker and optionally reload data.

        Args:
            ticker (str): New stock ticker symbol.
            reload (bool): Whether to reload data for the new ticker.
        """
        self.ticker = ticker
        if reload is True:
            self.load_data()
            self.loaded = True
        else:
            self.loaded = False

    def set_dates(self, start_date, end_date, reload = False):
        """
        Set a new date range and optionally reload data.

        Args:
            start_date (str): New start date (format: 'YYYY-MM-DD').
            end_date (str): New end date (format: 'YYYY-MM-DD').
            reload (bool): Whether to reload data for the new date range.
        """
        self.start_date = start_date
        self.end_date = end_date
        if reload is True:
            self.load_data()
            self.loaded = True
        else:
            self.loaded = False

    def get_ticker(self):
        """
        Get the current stock ticker.

        Returns:
            str: Current stock ticker symbol.
        """
        return self.ticker

    def get_dates(self):
        """
        Get the current date range.

        Returns:
            dict: Dictionary containing 'start_date' and 'end_date'.
        """
        return { "start_date":self.start_date, "end_date": self.end_date }

    def get_raw_data(self):
        """
        Get the loaded stock data.

        Returns:
            pd.DataFrame: DataFrame containing stock data.
        """
        return self.data

    def get_model_data(self):
        """
        Get the training and testing data.

        Returns:
            dict: Dictionary containing 'x_train', 'y_train', 'x_test', and 'y_test'.
        """
        return {'x_train': self.x_train, 'y_train': self.y_train, 'x_test': self.x_test, 'y_test': self.y_test }

    def load_data(self):
        """
        Load stock data for the specified ticker and date range.

        Returns:
            bool: True if data is loaded successfully, None otherwise.
        """
        try:
            df = yf.download(tickers=self.ticker,
                        start=self.start_date,
                        end=self.end_date,
                        auto_adjust=False)


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

            return True

        except Exception as e:

            print("Failure loading: ", self.ticker)

            print(e)

        return False

    def load_lob_data(self, filepath):
        """
        Load limit order book (LOB) data from a CSV file.

        Args:
            filepath (str): Path to the CSV file containing LOB data.

        Returns:
            bool: True if LOB data is loaded successfully, False otherwise.
        """
        try:
            lob_df = pd.read_csv(filepath)

            # Convert timestamp to datetime
            lob_df['timestamp'] = pd.to_datetime(lob_df['timestamp'], unit='s')

            # Set timestamp as the index
            lob_df.set_index('timestamp', inplace=True)

            # Ensure the index is sorted
            lob_df.sort_index(inplace=True)

            self.lob_data = lob_df
            print(f"LOB data loaded from {filepath}.")
            return True

        except Exception as e:
            print(f"Failed to load LOB data from {filepath}.")
            print(e)
            return False

#     def split_data(self):
#         """
#         Split the data into training and testing sets based on the split_date.

#         Raises:
#             ValueError: If features or target are not defined or if the resulting datasets are empty.
#         """
#         # Ensure the index and split_date are datetime objects
#         if not isinstance(self.data.index, pd.DatetimeIndex):
#             self.data.index = pd.to_datetime(self.data.index)
#         self.split_date = pd.to_datetime(self.split_date)

#         # Split the data
#         train_data = self.data[:self.split_date]
#         test_data = self.data[self.split_date:]

#         # Handle missing features or targets
#         if not self.features:
#             raise ValueError("Features are not defined. Ensure 'create_feature_list' is called.")
#         if not self.target:
#             raise ValueError("Target is not defined. Ensure 'create_target' is called.")

#         self.x_train = train_data[self.features]
#         self.x_test = test_data[self.features]
#         self.y_train = train_data[self.target]
#         self.y_test = test_data[self.target]

#         # Validate that the datasets are not empty
#         if self.x_train.empty or self.y_train.empty:
#             raise ValueError("Training dataset is empty. Check your feature and target creation.")
#         if self.x_test.empty or self.y_test.empty:
#             raise ValueError("Testing dataset is empty. Check your feature and target creation.")
#         return

#     def create_lagged_features(self, num_lags):
#         """
#         Create lagged features for the regression model.

#         Args:
#             num_lags (int): Number of lagged periods to create. Positive for past lags, negative for future lags.
#         """
#         if num_lags > 0:
#             for i in range(1, num_lags + 1):
#                 self.data[f'min_{i}_{self.feature_column}'] = self.data[self.feature_column].shift(i)
#         elif num_lags < 0:
#             self.data[f'fut_{-num_lags}_{self.feature_column}'] = self.data[self.feature_column].shift(num_lags)

#         self.data.dropna(inplace=True)  # Drop rows with NaN values created by shifting

#     def create_trend_features(self, num_lags):
#         """
#         Create lagged trend features for the regression model.

#         Args:
#             num_lags (int): Number of lagged periods to calculate trends.
#         """
#         self.create_lagged_features(num_lags + 1)

#         overall_trend = 0
#         for i in range(1, num_lags + 1):
#             self.data[f'min_{i}_trend'] = np.where(
#                 self.data[f'min_{i}_{self.feature_column}'] > self.data[f'min_{i+1}_{self.feature_column}'], 1, -1
#             )
#             overall_trend += self.data[f'min_{i}_trend']

#         self.data[f'trend_{num_lags}_day'] = np.where(overall_trend > 0, 1, -1)
#         self.data.dropna(inplace=True)  # Drop rows with NaN values created by shifting

#     def standardise_input(self, features):
#         """
#         Standardise the input features by subtracting the mean and dividing by the standard deviation.

#         Args:
#             features (list): List of feature column names to standardise.
#         """
#         for feature in features:
#             mu = float(self.x_train[feature].mean())
#             sigma = float(self.x_train[feature].std())
#             self.x_train[feature] = (self.x_train[feature] - mu) / sigma
#             self.x_test[feature] = (self.x_test[feature] - mu) / sigma

#     def calculate_log_returns(self):
# # !!! Need to make this use Feature column.
#         """
#         Calculate log returns for the close prices.

#         Raises:
#             ValueError: If 'close' column is not available in the data.
#         """
#         if 'close' in self.data.columns:
#             self.data['log_returns'] = np.log(self.data['close'] / self.data['close'].shift(1))
#             self.data.dropna(inplace=True)  # Remove rows with NaN values caused by shifting
#             self.feature_column = 'log_returns'  # Update feature column to log returns
#         else:
#             raise ValueError("Close prices are not available in the data.")

#     def create_feature_list(self, comparison):
#         """
#         Generate the list of required features based on the comparison details.

#         Args:
#             comparison (dict): Dictionary containing model and parameter details.

#         Returns:
#             list: List of feature column names.
#         """
#         model = comparison['model']
#         parameters = comparison['params']

#         # Handle log returns if enabled in the comparison
#         if comparison.get('use_log_returns', False):
#             self.calculate_log_returns()

#         # Handle lagged features
#         num_days_lag = int(parameters.get('num_days_lag', {}).get('value', 0))
#         if num_days_lag > 0:
#             self.create_lagged_features(num_days_lag)

#         # Handle trend features for models that require them
#         if model == 'LinearRegression' and 'trend_features' in parameters:
#             trend_days = int(parameters.get('trend_features', {}).get('value', 0))
#             if trend_days > 0:
#                 self.create_trend_features(trend_days)

#         self.features = [col for col in self.data.columns if col.startswith('min_') or col.startswith('trend_')]
#         if not self.features:
#             raise ValueError("No features were created. Check your feature creation logic.")
#         return self.features

#     def create_target(self, comparison):
#         """
#         Generate the target based on the comparison details.

#         Args:
#             comparison (dict): Dictionary containing model and parameter details.

#         Returns:
#             list: List of target column names.
#         """
#         parameters = comparison['params']
#         forward_projection_days = int(parameters.get('forward_projection_days', {}).get('value', 0))

#         if forward_projection_days > 1:
#             self.create_lagged_features(-forward_projection_days)

#         if forward_projection_days == 1:
#             self.target = [self.feature_column]
#         else:
#             self.target = [f'fut_{forward_projection_days}_{self.feature_column}']

#         if not self.target:
#             raise ValueError("No target was created. Check your target creation logic.")
#         return self.target

#     def process_comparison(self, comparison):
# # Review inputs and outputs
#         """
#         Process a comparison by creating features, targets, and splitting the data.

#         Args:
#             comparison (dict): The comparison details containing model, parameters, etc.
#         """
#         # Handle log returns if enabled in the comparison
#         if comparison.get('use_log_returns', False):
#             self.calculate_log_returns()

#         # Create features and target based on the comparison
#         self.create_feature_list(comparison)
#         self.create_target(comparison)

#         # Split the data into training and testing sets
#         self.split_data()

