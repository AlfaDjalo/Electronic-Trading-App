""" Data object for time series for a stock """

import pandas as pd
import numpy as np
import yfinance as yf
# import statsmodels.api as sm
# import matplotlib.pyplot as plt

# Global constants for default dates
DEFAULT_START_DATE = '2014-12-31'
DEFAULT_END_DATE = '2024-12-31'

# Global variable for LOB data filepath
LOB_FILEPATH = "lob_data/order_book_history.csv"

class StockData:
    def __init__(self, data_type, ticker, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE, load_data=False):
    # def __init__(self, ticker, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE, split_date='2022-12-31', load_data=False, create_model_data=False, use_lob_data=False):
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
        """
        self.data_type = data_type
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

        # self.split_date = split_date
        # self.x_train = None
        # self.y_train = None
        # self.x_test = None
        # self.y_test = None
        # self.feature_column = 'close'
        # self.features = []
        # self.target = []

        # self.use_lob_data = use_lob_data
        # self.lob_data = None  # Initialize LOB data attribute
        # self.data_type = "intraday" if self.use_lob_data else "daily"  # Set data_type based on use_lob_data

        if load_data is True:
            if self.data_type == "intraday":
                if LOB_FILEPATH:
                    self.load_lob_data(LOB_FILEPATH)
                    self.loaded = True
                else:
                    raise ValueError("LOB data file path must be provided for intraday data.")
            else:
                self.load_daily_data()
                self.loaded = True
        else:
            self.loaded = False

        # if load_data is True:
        #     if self.use_lob_data:
        #         if LOB_FILEPATH:
        #             self.load_lob_data(LOB_FILEPATH)
        #             self.loaded = True
        #         else:
        #             raise ValueError("LOB data file path must be provided when use_lob_data is True.")
        #     else:
        #         self.load_data()
        #         self.loaded = True
        # else:
        #     self.loaded = False

        # if create_model_data is True and self.loaded is True:
        #     self.split_data() # !!! Need to use full
        #     self.model_data_created = True
        # else:
        #     self.model_data_created = False

    def __repr__(self):
        """
        Representation of StockData object
        
        """
        return self.data.head(5)

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

    def get_data_type(self):
        """
        Get the current data type.

        Returns:
            str: Current data type.
        """
        return self.data_type

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

    def get_data(self):
        """
        Get the loaded stock data.

        Returns:
            pd.DataFrame: DataFrame containing stock data.
        """
        if self.data is None:
            raise ValueError("Stock data is not loaded.")
        return self.data

    # def get_raw_data(self):
    #     """
    #     Get the loaded stock data.

    #     Returns:
    #         pd.DataFrame: DataFrame containing stock data.
    #     """
    #     return self.data

    # def get_model_data(self):
    #     """
    #     Get the training and testing data.

    #     Returns:
    #         dict: Dictionary containing 'x_train', 'y_train', 'x_test', and 'y_test'.
    #     """
    #     return {'x_train': self.x_train, 'y_train': self.y_train, 'x_test': self.x_test, 'y_test': self.y_test }

    def load_daily_data(self):
        """
        Load stock data for the specified ticker and date range.

        Returns:
            bool: True if data is loaded successfully, None otherwise.
        """
        print(f"Loading data for {self.ticker} from {self.start_date} to {self.end_date}...")
        if not self.ticker:
            print("No ticker provided.")
            return False
        try:
            df = yf.download(tickers=self.ticker,
                             start=self.start_date,
                             end=self.end_date,
                             auto_adjust=False)

            if df.empty:
                raise ValueError(f"No data found for ticker {self.ticker} in the specified date range.")

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
                print('This is {:.3f}% of the total.'.format(nof_missing_values * 100 / len(df)))

                df['close'] = df['close'].bfill()
                nof_missing_values = sum(np.isnan(df['close']))
                print('Now', nof_missing_values, 'observations are missing.')

            return True

        except ValueError as e:
            print(f"ValueError: {e}")
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
        print(f"Loading LOB data from {filepath}...")
        if not filepath:
            print("No LOB data file provided.")
            return False
        try:
            lob_df = pd.read_csv(filepath)

            if lob_df.empty:
                raise ValueError(f"LOB data file {filepath} is empty.")

            # Convert timestamp to datetime
            lob_df['timestamp'] = pd.to_datetime(lob_df['timestamp'], unit='s')

            # Set timestamp as the index
            lob_df.set_index('timestamp', inplace=True)

            # Ensure the index is sorted
            lob_df.sort_index(inplace=True)

            # Add mid_price column
            if 'bid_price_0' in lob_df.columns and 'ask_price_0' in lob_df.columns:
                lob_df['mid_price'] = (lob_df['bid_price_0'] + lob_df['ask_price_0']) / 2
            else:
                print("Missing 'bid_price_0' or 'ask_price_0' columns. Cannot calculate 'mid_price'.")

            self.data = lob_df
            print(f"LOB data loaded from {filepath}.")
            print(self.data.head())

            return True

        except ValueError as e:
            print(f"ValueError: {e}")
        except Exception as e:
            print(f"Failed to load LOB data from {filepath}.")
            print(e)
            return False

    # def get_lob_data(self):
    #     """
    #     Get the loaded LOB data.

    #     Returns:
    #         pd.DataFrame: DataFrame containing LOB data.
    #     """
    #     if self.lob_data is None:
    #         raise ValueError("LOB data is not loaded.")
    #     return self.lob_data

    # def get_daily_data(self):
    #     """
    #     Get the loaded daily stock data.

    #     Returns:
    #         pd.DataFrame: DataFrame containing daily stock data.
    #     """
    #     if self.data is None:
    #         raise ValueError("Daily stock data is not loaded.")
    #     return self.data

    def get_available_fields(self):
        """
        Get available fields based on the data type.

        Args:
            data_type (str): The type of data ('daily' or 'intraday').

        Returns:
            list: A list of available fields.
        """
        return list(self.data.columns)

    # def get_available_fields(self):
    #     """
    #     Get available fields based on the data type.

    #     Args:
    #         data_type (str): The type of data ('daily' or 'intraday').

    #     Returns:
    #         list: A list of available fields.
    #     """
    #     print(self.data_type)
    #     if self.data_type == 'daily':
    #     # if self.data_type == 'daily' and self.data is not None:
    #         return list(self.data.columns)
    #     elif self.data_type == 'intraday':
    #     # elif self.data_type == 'intraday' and self.lob_data is not None:
    #         return list(self.lob_data.columns)
    #     else:
    #         return []

    # def get_data_type(self):
    #     """
    #     Get the data type (daily or intraday).

    #     Returns:
    #         str: The data type of the stock data.
    #     """
    #     return self.data_type

