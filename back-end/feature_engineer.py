import pandas as pd
import numpy as np
import ta

class FeatureEngineer:
    def __init__(self, raw_df, feature_set_config):
        print("Creating FeatureEngineer.")
        # print(feature_set_config)
        self.data = raw_df.copy()
        self.config = feature_set_config
    
    def apply(self):
        print("Applying FeatureEngineer.")
        # print(self.config)
        for feature in self.config["features"]:
            print(feature)
            name = feature["name"]
            input_data_fields = feature["input_data_fields"]
            function = feature.get("function", "raw_data")
            function_parameters = feature.get("function_parameters", {})

            if not isinstance(input_data_fields, list):
                raise ValueError(f"input_data_fields for feature '{name}' must be a list.")

            for field in input_data_fields:
                if field not in self.data.columns:
                    raise ValueError(f"Field '{field}' is missing in the data.")


            # Find function
            method = getattr(self, function, None)
            if not method:
                raise ValueError(
                    f"Unsupported function '{function}' in feature set. Ensure it is implemented in MLData."
                )

            # print(f"Running feature: {name} with function {function}")
            # result = method(input_data_fields, **function_parameters) if function_parameters else method(input_data_fields)
            # print("Result type:", type(result))
            # self.data[name] = result
            # print(f"Feature '{name}' created successfully.")
            
            # Compute feature column

            # if function == "raw_data":
            #         self.data[name] = 
            # else:
            if function_parameters:
                self.data[name] = method(input_data_fields, **function_parameters)
            else:
                self.data[name] = method(input_data_fields)
            print(f"Feature '{name}' created successfully.")

            print(self.data.head(5))
            # Track features/target
            # if name.lower() == "target":
            #     self.target = name
            # else:
            #     self.features.append(name)

            # if self.verbose:

        # print(self.data.head(5))

        # Clean up
        # self.data.dropna(inplace=True)

        return self.data
    # except Exception as e:
    #     raise RuntimeError(f"Error applying feature set: {e}")


    def create_lag(self, df, lag=1):
        df[f"lag_{lag}"] = df["mid_price"].shift(lag)
        return df
    
    def create_rsi(self, df, window=14):
        delta = df["mid_price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        return df


    def create_lag(self, input_field, num_lags=1, **kwargs):
        """
        Create lagged features for the specified input field.

        Args:
            input_field (str): The column name to create lagged features for.
            num_lags (int): Number of lagged periods to create.
            **kwargs: Additional arguments (not used here).
        """
        try:
            input_field = input_field[0]
            
            return self.data[input_field].shift(num_lags)

        except KeyError:
            raise ValueError(f"Feature column '{input_field}' is missing in the data.")
        except Exception as e:
            raise RuntimeError(f"Error during lagged feature creation: {e}")

    def create_average(self, input_fields):
        """
        Calculate the average of the specified input fields.

        Args:
            input_fields (list): List of column names to calculate the average for.

        Returns:
            pd.Series: A series containing the average of the input fields.
        """
        try:
            # Ensure all input fields exist in the data
            for field in input_fields:
                if field not in self.data.columns:
                    raise ValueError(f"Field '{field}' is missing in the data.")

            # Calculate the average across the specified fields
            return self.data[input_fields].mean(axis=1)

        except Exception as e:
            raise RuntimeError(f"Error during average calculation: {e}")

    def create_moving_average(self, input_fields, window, alpha=1.0):
        """
        Calculate a moving average of the specified input field.

        Args:
            input_field: The column name to create the moving average for.
            window (int): Number of lagged periods to create.
            exponential_weight (float): Weighting for exponentially weighted moving average.
            **kwargs: Additional arguments (not used here).

        Returns:
            pd.Series: A series containing the average of the input fields.
        """
        series = self.data[input_fields[0]]

        if alpha == 1.0:
            # Use ta library's SMA for consistency with your other code
            return ta.trend.sma_indicator(series, window=window)
        else:
            # Custom-alpha EWMA (ta does not expose custom alpha, so use pandas)
            return series.ewm(alpha=alpha, adjust=False).mean()

    def create_lagged_average(self, input_fields, num_lags=1, **kwargs):
        """
        Calculate the average of the specified input fields
        then apply a lag.

        Args:
            input_fields (list): List of column names to calculate the average for.
            num_lags (int): Number of lagged periods to create.
            **kwargs: Additional arguments (not used here).

        Returns:
            pd.Series: A series containing the average of the input fields.
        """
        print("Creating lagged average")
        try:
            # Ensure all input fields exist in the data
            for field in input_fields:
                if field not in self.data.columns:
                    raise ValueError(f"Field '{field}' is missing in the data.")

            # Calculate the average across the specified fields
            temp_field =  self.data[input_fields].mean(axis=1)
        
            print("Creating average, about to lag")
            return temp_field.shift(num_lags)
        
        except Exception as e:
            raise RuntimeError(f"Error during average calculation: {e}")

    def create_rsi(self, input_field, window=20):
        """
        Calculate the RSI (Relative Strength Index) for one field or
        for the average of multiple fields.

        Args:
            input_field (str or list): Column name(s) to calculate RSI on.
            window (int): RSI calculation window.

        Returns:
            pd.Series: RSI values.
        """

        # Case 1: single field (string)
        if isinstance(input_field, str):
            series_data = self.data[input_field]

        # Case 2: multiple fields (list)
        elif isinstance(input_field, (list, tuple)):
            series_data = self.create_average(input_field)

        else:
            raise ValueError("input_field must be a string or list of strings.")

        return ta.momentum.RSIIndicator(close=series_data, window=window).rsi()

    def create_rsi_old(self, input_field, window=20):
        """
        Calculate the RSI (Relative Strength Indicator) of the specified input fields.

        Args:
            input_fields (str): Name of column to calculate the rsi for.
            Window (integer): The window which the rsi is calculated over.

        Returns:
            pd.Series: A series containing the rsi for the input field.
        """
        data_col = self.data[input_field]

        if isinstance(data_col, pd.DataFrame):
            series_data = data_col.iloc[:,0]
        else:
            series_data = data_col

        return ta.momentum.RSIIndicator(close=series_data, window=window).rsi()

    def create_bb_high(self, input_field, window=20):
        """
        Calculate the High Bollinger Bands of the specified input field.

        Args:
            input_fields (str): Name of column to calculate the metric for.
            Window (integer): The window which the metric is calculated over.

        Returns:
            pd.Series: A series containing the metric for the input field.
        """
        data_col = self.data[input_field]

        if isinstance(data_col, pd.DataFrame):
            series_data = data_col.iloc[:,0]
        else:
            series_data = data_col

        return ta.volatility.BollingerBands(close=np.log1p(series_data), window=window).bollinger_hband()

    def create_bb_mid(self, input_field, window=20):
        """
        Calculate the Mid Bollinger Band of the specified input fields.

        Args:
            input_fields (str): Name of column to calculate the metric for.
            Window (integer): The window which the metric is calculated over.

        Returns:
            pd.Series: A series containing the metric for the input field.
        """
        data_col = self.data[input_field]

        if isinstance(data_col, pd.DataFrame):
            series_data = data_col.iloc[:,0]
        else:
            series_data = data_col

        return ta.volatility.BollingerBands(close=np.log1p(series_data), window=window).bollinger_mavg()

    def create_bb_low(self, input_field, window=20):
        """
        Calculate the Low Bollinger Band of the specified input fields.

        Args:
            input_fields (str): Name of column to calculate the metric for.
            Window (integer): The window which the metric is calculated over.

        Returns:
            pd.Series: A series containing the metric for the input field.
        """
        data_col = self.data[input_field]

        if isinstance(data_col, pd.DataFrame):
            series_data = data_col.iloc[:,0]
        else:
            series_data = data_col

        return ta.volatility.BollingerBands(close=np.log1p(series_data), window=window).bollinger_lband()

    def create_macd(self, input_field, window_slow=26, window_fast=12, window_sign=9):
        """
        Calculate the macd (Moving Average Convergence Divergence ?) of the specified input fields.

        Args:
            input_fields (str): Name of column to calculate the metric for.
            Window (integer): The window which the metric is calculated over.

        Returns:
            pd.Series: A series containing the metric for the input field.
        """
        data_col = self.data[input_field]

        if isinstance(data_col, pd.DataFrame):
            series_data = data_col.iloc[:,0]
        else:
            series_data = data_col

        macd = ta.trend.MACD(close=series_data, window_slow=window_slow, window_fast=window_fast, window_sign=window_sign).macd()

        return macd.sub(macd.mean()).div(macd.std())

    def raw_data(self, input_field):
        """
        Return the input field unchanged.

        Args:
            input_field (str): The column name to return unchanged.
            **kwargs: Additional arguments (not used here).

        Returns:
            pd.Series: The unchanged column.
        """
        if (input_field[0] not in self.data.columns):
            raise ValueError(f"Field '{input_field[0]}' is missing in the data.")
        return self.data[input_field[0]]
