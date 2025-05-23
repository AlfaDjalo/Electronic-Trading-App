import pandas as pd
import numpy as np
import json
import ta

FEATURE_SETS_FILE = "c:\\Users\\David\\Projects\\Electronic Trading App\\data\\feature_sets.json"

class MLData:
    def __init__(self, raw_data, train_percentage=0.8, feature_set=None, normalise=False):
        """
        Initialise MLData object.

        Args:
            raw_data (pd.DataFrame): DataFrame containing raw data (daily or intraday).
            train_percentage (float): Percentage of data to use for training (0 < train_percentage < 1).
            feature_set (str): Feature set to apply.
            normalise (bool): Whether to normalise the input features.
        """
        if raw_data is None:
            raise ValueError("Raw data cannot be None. Ensure StockData is properly loaded.")

        if not (0 < train_percentage < 1):
            raise ValueError("train_percentage must be a float between 0 and 1.")

        self.data = raw_data.copy()
        self.train_percentage = train_percentage
        self.features = []
        self.target = None
        self.feature_set = feature_set
        self.normalise = normalise

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.normalisation_params = {}  # Dictionary to store mean and std for each column

        self.process_data()

    def __repr__(self):
        """
        Representation of MLData object
        
        """
        return self.data.head(5)

    def process_data(self):
        """
        Process the data based on the feature set and apply transformations.
        """
        try:
            if self.feature_set:
                self.apply_feature_set(self.feature_set)

            self.split_data()

            # Apply normalization if enabled
            if self.normalise:
                self.normalise_input(self.features)
                self.normalise_target()

        except Exception as e:
            raise RuntimeError(f"Error during data processing: {e}")

    def split_data(self):
        """
        Split the data into training and testing sets based on the train_percentage.

        Raises:
            ValueError: If features or target are not defined or if the resulting datasets are empty.
        """
        try:
            train_size = int(len(self.data) * self.train_percentage)
            train_data = self.data.iloc[:train_size]
            test_data = self.data.iloc[train_size:]

            # Validate features and target
            if not isinstance(self.features, list) or not all(isinstance(f, str) for f in self.features):
                raise ValueError("Features must be a list of strings.")
            if not isinstance(self.target, str):
                raise ValueError("Target must be a string.")

            if not self.features:
                raise ValueError("Features are not defined. Ensure 'create_lagged_features' is called.")
            if not self.target:
                raise ValueError("Target is not defined. Ensure 'create_target' is called.")

            # Ensure features and target exist in the data
            missing_features = [f for f in self.features if f not in self.data.columns]
            if missing_features:
                raise ValueError(f"Missing features in data: {missing_features}")
            if self.target not in self.data.columns:
                raise ValueError(f"Target column '{self.target}' is missing in data.")

            self.x_train = train_data[self.features]
            self.x_test = test_data[self.features]
            self.y_train = train_data[[self.target]]
            self.y_test = test_data[[self.target]]

            # Validate that the datasets are not empty
            if self.x_train.empty or self.y_train.empty:
                raise ValueError("Training dataset is empty. Check your feature and target creation.")
            if self.x_test.empty or self.y_test.empty:
                raise ValueError("Testing dataset is empty. Check your feature and target creation.")
        except Exception as e:
            raise RuntimeError(f"Error during data splitting: {e}")

    def create_lagged_features(self, input_field, num_lags=1, **kwargs):
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

    def create_rsi(self, input_field, window=20):
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


    def normalise_columns(self, columns, train_data, test_data):
        """
        Normalise the specified columns by subtracting the mean and dividing by the standard deviation.

        Args:
            columns (list): List of column names to normalise.
            train_data (pd.DataFrame): Training data.
            test_data (pd.DataFrame): Testing data.
        """
        try:
            for column in columns:
                if column not in self.normalisation_params:
                    mu = float(train_data[column].mean())
                    sigma = float(train_data[column].std())
                    self.normalisation_params[column] = {"mean": mu, "std": sigma}
                else:
                    mu = self.normalisation_params[column]["mean"]
                    sigma = self.normalisation_params[column]["std"]

                train_data[column] = (train_data[column] - mu) / sigma
                test_data[column] = (test_data[column] - mu) / sigma
        except Exception as e:
            raise RuntimeError(f"Error during column normalisation: {e}")

    def apply_normalisation_to_series(self, series, column_name):
        """
        Apply stored normalisation parameters to a new series.

        Args:
            series (pd.Series): The series to normalise.
            column_name (str): The name of the column to retrieve normalisation parameters for.

        Returns:
            pd.Series: The normalised series.
        """
        if column_name not in self.normalisation_params:
            raise ValueError(f"No normalisation parameters found for column '{column_name}'.")
        params = self.normalisation_params[column_name]
        return (series - params["mean"]) / params["std"]

    def normalise_input(self, columns):
        """
        Normalise the input columns (features and target) by calling `normalise_columns`.

        Args:
            columns (list): List of column names to normalise.
        """
        self.normalise_columns(columns, self.x_train, self.x_test)

    def normalise_target(self):
        """
        Normalise the target column by calling `normalise_columns`.
        """
        if self.target:
            self.normalise_columns([self.target], self.y_train, self.y_test)

    def calculate_log_returns(self, input_field, **kwargs):
        """
        Calculate log returns for the specified input field.

        Args:
            input_field (str): The column name to calculate log returns for.
        """
        try:
            if input_field in self.data.columns:
                self.data[f'{input_field}_log_returns'] = np.log(self.data[input_field] / self.data[input_field].shift(1))
                self.data.dropna(inplace=True)
            else:
                raise ValueError(f"{input_field} is not available in the data.")
        except Exception as e:
            raise RuntimeError(f"Error during log return calculation: {e}")

    # Getters
    # def get_feature_column(self):
    #     return self.feature_column

    # def get_split_date(self):
    #     return self.split_date

    # def get_features(self):
    #     return self.features

    def get_target(self):
        return self.target

    def get_log_returns(self):
        return self.log_returns

    def get_normalise(self):
        return self.normalise

    def get_data(self):
        """
        Get the processed data.

        Returns:
            dict: Dictionary containing 'x_train', 'y_train', 'x_test', and 'y_test'.
        """
        return {
            'x_train': self.x_train,
            'y_train': self.y_train,
            'x_test': self.x_test,
            'y_test': self.y_test
        }

    def get_normalisation_params(self, feature_name):
        """
        Get the normalisation parameters (mean and std) for a specific feature.

        Args:
            feature_name (str): The name of the feature.

        Returns:
            dict: A dictionary containing 'mean' and 'std' for the feature.
        """
        if feature_name not in self.normalisation_params:
            raise ValueError(f"No normalisation parameters found for feature '{feature_name}'.")
        return self.normalisation_params[feature_name]

    def set_log_returns(self, log_returns):
        if not isinstance(log_returns, bool):
            raise TypeError("Log returns must be a boolean.")
        self.log_returns = log_returns
        self.process_data()

    def set_normalise(self, normalise):
        if not isinstance(normalise, bool):
            raise TypeError("normalise must be a boolean.")
        self.normalise = normalise
        self.process_data()

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

    def apply_feature_set(self, feature_set):
        """
        Apply a predefined feature set to the data.

        Args:
            feature_set (dict): Feature set configuration containing feature definitions.
        """
        try:
            # feature_set = self.load_feature_set()

            for feature in feature_set.get("features", []):
                name = feature["name"]
                input_data_fields = feature["input_data_fields"]
                function = feature["function"]
                function_parameters = feature.get("function_parameters", {})

                # Ensure input_data_fields is a list and process each field
                if not isinstance(input_data_fields, list):
                    raise ValueError(f"input_data_fields for feature '{name}' must be a list.")

                for field in input_data_fields:
                    if field not in self.data.columns:
                        raise ValueError(f"Field '{field}' is missing in the data.")

                # Dynamically call the corresponding method
                # print(f"Processing feature: {name}, fields: {input_data_fields}, function: {function}, params: {function_parameters}")
                method = getattr(self, function, None)
                if not method:
                    raise ValueError(f"Unsupported function '{function}' in feature set. Ensure it is implemented in MLData.")

                # Pass the list of fields as arguments to the method
                if function_parameters:
                    self.data[name] = method(input_data_fields, **function_parameters)
                else:
                    self.data[name] = method(input_data_fields)
                # print(f"Feature '{name}' created successfully.")

                self.features.append(name)

            target = feature_set.get("target", {})
            if target:
                name = "target"
                input_data_fields = target["input_data_fields"]
                function = target["function"]
                function_parameters = target.get("function_parameters", {})

                # Ensure input_data_fields is a list and process each field
                if not isinstance(input_data_fields, list):
                    raise ValueError(f"input_data_fields for '{name}' must be a list.")

                for field in input_data_fields:
                    if field not in self.data.columns:
                        raise ValueError(f"Field '{field}' is missing in the data.")

                # Dynamically call the corresponding method
                # print(f"Processing target: fields: {input_data_fields}, function: {function}, params: {function_parameters}")
                method = getattr(self, function, None)
                if not method:
                    raise ValueError(f"Unsupported function '{function}' in target configuration. Ensure it is implemented in MLData.")

                # Pass the list of fields as arguments to the method
                self.data[name] = method(input_data_fields, **function_parameters)
                self.target = name
                # print(f"Target '{name}' created successfully.")

            self.data.dropna(inplace=True)
        except Exception as e:
            raise RuntimeError(f"Error applying feature set: {e}")

    def get_feature_set(self):
        """
        Get the current feature set or feature set name.

        Returns:
            dict or str: The feature set dictionary if set, otherwise the feature set name.
        """
        return self.feature_set if self.feature_set else self.feature_set_name

    def set_feature_set(self, feature_set=None, feature_set_name=None):
        """
        Set the feature set directly or by name.

        Args:
            feature_set (dict): The feature set dictionary to apply.
            feature_set_name (str): The name of the feature set to load.
        """
        if feature_set and feature_set_name:
            self.feature_set = feature_set
            self.feature_set_name = feature_set_name
        else:
            raise ValueError("Either feature_set or feature_set_name must be provided.")
        self.process_data()
