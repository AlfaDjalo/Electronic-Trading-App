import pandas as pd
import numpy as np

class MLData:
    def __init__(self, raw_data, lag_period, forecast_period, features=None, target=None, 
                 split_date='2022-12-31', feature_column='close', log_returns=False, standardised=False):
        """
        Initialise MLData object.

        Args:
            stock_data (StockData): StockData object containing raw data.
            lag_period (int): Number of lagged periods for features.
            forecast_period (int): Number of periods for forecasting (negative lag).
            features (list): List of feature column names.
            target (str): Target column name.
            split_date (str): Date to split data into training and testing sets.
            feature_column (str): Column to be used as the main feature.
            log_returns (bool): Whether to calculate log returns for the feature column.
            standardised (bool): Whether to standardise the input features.
        """
        # self.data = stock_data.data.copy()
        self.data = raw_data.copy()
        self.feature_column = feature_column
        self.split_date = pd.to_datetime(split_date)
        self.lag_period = lag_period
        self.forecast_period = forecast_period
        self.features = features if features else raw_data.columns.tolist()
        self.target = target if target else (
            f'fut_{forecast_period}_{feature_column}' if forecast_period != 1 else feature_column
        )
        self.log_returns = log_returns
        self.standardised = standardised

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.process_data()

    def process_data(self):
        try:
            if self.log_returns:
                self.calculate_log_returns()
            # print(self.data)
            self.create_lagged_features(self.lag_period)
            self.features = [col for col in self.data.columns if col.startswith(f'min_') and self.feature_column in col]
            self.create_lagged_features(-self.forecast_period)
            self.target = f'fut_{self.forecast_period}_{self.feature_column}'
            # Ensure the target column is created before splitting and standardization
            if self.target not in self.data.columns:
                raise ValueError(f"Target column '{self.target}' is missing in the data.")
            self.split_data()

            if self.standardised:
                self.standardise_input(self.features + [self.target])  # Include target in standardization
        except Exception as e:
            raise RuntimeError(f"Error during data processing: {e}")

    def split_data(self):
        """
        Split the data into training and testing sets based on the split_date.

        Raises:
            ValueError: If features or target are not defined or if the resulting datasets are empty.
        """
        try:
            if not isinstance(self.data.index, pd.DatetimeIndex):
                self.data.index = pd.to_datetime(self.data.index)

            train_data = self.data[:self.split_date]
            test_data = self.data[self.split_date:]

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

    def create_lagged_features(self, num_lags):
        """
        Create lagged features for the regression model.

        Args:
            num_lags (int): Number of lagged periods to create. Positive for past lags, negative for future lags.
        """
        try:
            if num_lags > 0:
                for i in range(1, num_lags + 1):
                    self.data[f'min_{i}_{self.feature_column}'] = self.data[self.feature_column].shift(i)
            elif num_lags < 0:
                self.data[f'fut_{-num_lags}_{self.feature_column}'] = self.data[self.feature_column].shift(num_lags)

            self.data.dropna(inplace=True)
        except Exception as e:
            raise RuntimeError(f"Error during lagged feature creation: {e}")

    def standardise_input(self, columns):
        """
        Standardise the input columns by subtracting the mean and dividing by the standard deviation.

        Args:
            columns (list): List of column names to standardise.
        """
        try:
            for column in columns:
                mu = float(self.x_train[column].mean())
                sigma = float(self.x_train[column].std())
                self.x_train[column] = (self.x_train[column] - mu) / sigma
                self.x_test[column] = (self.x_test[column] - mu) / sigma
        except Exception as e:
            raise RuntimeError(f"Error during input standardisation: {e}")

    def calculate_log_returns(self):
        """
        Calculate log returns for the feature column.

        Raises:
            ValueError: If the feature column is not available in the data.
        """
        try:
            if self.feature_column in self.data.columns:
                self.data['log_returns'] = np.log(self.data[self.feature_column] / self.data[self.feature_column].shift(1))
                self.data.dropna(inplace=True)
                self.feature_column = 'log_returns'
            else:
                raise ValueError(f"{self.feature_column} is not available in the data.")
        except Exception as e:
            raise RuntimeError(f"Error during log return calculation: {e}")

    # Getters
    def get_feature_column(self):
        return self.feature_column

    def get_split_date(self):
        return self.split_date

    def get_lag_period(self):
        return self.lag_period

    def get_forecast_period(self):
        return self.forecast_period

    def get_features(self):
        return self.features

    def get_target(self):
        return self.target

    def get_log_returns(self):
        return self.log_returns

    def get_standardised(self):
        return self.standardised

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

    # Setters with error checking
    def set_feature_column(self, feature_column):
        if not isinstance(feature_column, str):
            raise TypeError("Feature column must be a string.")
        self.feature_column = feature_column
        self.process_data()

    def set_split_date(self, split_date):
        try:
            self.split_date = pd.to_datetime(split_date)
        except Exception:
            raise ValueError("Split date must be a valid date string.")
        self.process_data()

    def set_lag_period(self, lag_period):
        if not isinstance(lag_period, int) or lag_period < 0:
            raise ValueError("Lag period must be a non-negative integer.")
        self.lag_period = lag_period
        self.process_data()

    def set_forecast_period(self, forecast_period):
        if not isinstance(forecast_period, int) or forecast_period < 0:
            raise ValueError("Forecast period must be a non-negative integer.")
        self.forecast_period = forecast_period
        self.process_data()

    def set_features(self, features):
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise TypeError("Features must be a list of strings.")
        self.features = features
        self.process_data()

    def set_target(self, target):
        if not isinstance(target, str):
            raise TypeError("Target must be a string.")
        self.target = target
        self.process_data()

    def set_log_returns(self, log_returns):
        if not isinstance(log_returns, bool):
            raise TypeError("Log returns must be a boolean.")
        self.log_returns = log_returns
        self.process_data()

    def set_standardised(self, standardised):
        if not isinstance(standardised, bool):
            raise TypeError("Standardised must be a boolean.")
        self.standardised = standardised
        self.process_data()
