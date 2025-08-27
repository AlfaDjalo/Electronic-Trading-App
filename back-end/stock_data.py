""" Data object for time series for a stock """

import pandas as pd
import numpy as np
import yfinance as yf

# Global constants for default dates
DEFAULT_START_DATE = '2014-12-31'
DEFAULT_END_DATE = '2024-12-31'

# Global variable for LOB data filepath
LOB_FILEPATH = "../data/order_book_history.csv"

class StockData:
    """
    Object storing the raw time series data for the selected stock
    """
    def __init__(self, data_type, ticker, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE, load_data=False, verbose=False):
        """
        Initialise StockData object.

        Args:
            datatype (str): Type of data - "daily" or "intraday"
            ticker (str): Stock ticker symbol.
            start_date (str): Start date for the data (format: 'YYYY-MM-DD').
            end_date (str): End date for the data (format: 'YYYY-MM-DD').
            load_data (bool): Whether to load stock data immediately.
        """
        self.verbose = verbose
        if self.verbose == True:
            print("Initializing StockData object")

        self.data_type = data_type
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

        if load_data is True:
            match self.data_type:
                case "intraday":
                    if LOB_FILEPATH:
                        self.load_lob_data(LOB_FILEPATH)
                        self.loaded = True
                    else:
                        raise ValueError("LOB data file path must be provided for intraday data.")
                case "daily":
                    self.load_daily_data()
                    self.loaded = True
                case "test":
                    self.load_test_data()
                    self.load = True
        else:
            self.loaded = False

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

    def load_daily_data(self):
        """
        Load stock data for the specified ticker and date range.

        Returns:
            bool: True if data is loaded successfully, None otherwise.
        """
        if self.verbose == True:
            print(f"Loading daily data for {self.ticker}.")

        if not self.ticker:
            print("No ticker provided.")
            return False
        try:
            print(f"Loading data for {self.ticker} from {self.start_date} to {self.end_date}...")
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
        if self.verbose == True:
            print(f"Loading intraday data for {self.ticker}.")

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
            # print(f"LOB data loaded from {filepath}.")
            # print(self.data.head())

            return True

        except ValueError as e:
            print(f"ValueError: {e}")
        except Exception as e:
            print(f"Failed to load LOB data from {filepath}.")
            print(e)
            return False

    def load_test_data(self):
        """
        Create test time series data in the same format as Yahoo Finance data.
        
        Returns:
            bool: True if data is created successfully, False otherwise.
        """
        if self.verbose == True:
            print(f"Loading test data for {self.ticker}.")

        valid_series = ['flat', 'ramp', 'wave']
        if self.ticker not in valid_series:
            print(f"Invalid series type. Choose from: {valid_series}")
            return False
            
        try:
            # print(f"Creating {self.ticker} test data from {self.start_date} to {self.end_date}...")
            
            # Create date range
            date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            n_days = len(date_range)
            
            if n_days == 0:
                raise ValueError("Invalid date range - no days generated.")
            
            # Generate close prices based on series type
            if self.ticker == 'flat':
                close_prices = np.full(n_days, 100.0)
                
            elif self.ticker == 'ramp':
                close_prices = np.linspace(50, 150, n_days)
                
            elif self.ticker == 'wave':
                # One cycle per month (roughly 30 days)
                # Sine wave oscillating between 50 and 150
                t = np.arange(n_days)
                frequency = 2 * np.pi / 30  # One cycle per 30 days
                close_prices = 100 + 50 * np.sin(frequency * t)  # Center at 100, amplitude 50
            
            # Create DataFrame with same structure as Yahoo Finance data
            df = pd.DataFrame(index=date_range)
            df.index.name = 'date'
            
            # Add all the typical Yahoo Finance columns, but only populate 'close'
            # df['open'] = np.nan
            # df['high'] = np.nan  
            # df['low'] = np.nan
            # df['close'] = close_prices
            # df['adj close'] = np.nan
            # df['volume'] = np.nan
            df['feature1'] = close_prices
            
            self.data = df
            
            # print(f"{self.ticker} test series loaded.")
            # print(f"Generated {n_days} data points.")
                            
            return True
            
        except ValueError as e:
            print(f"ValueError: {e}")
        except Exception as e:
            print(f"Failure creating {self.ticker} test data:")
            print(e)
            
        return False


    def get_available_fields(self):
        """
        Get available fields based on the data type.

        Args:
            data_type (str): The type of data ('daily' or 'intraday').

        Returns:
            list: A list of available fields.
        """
        return list(self.data.columns)