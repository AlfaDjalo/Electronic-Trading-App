"""Data processing pipeline for time series modeling.

This module defines the `DataProcessor` class, which orchestrates the
end-to-end preparation of time series data for machine learning. The
pipeline includes:
    - Feature engineering
    - Train/validation/test splitting
    - Normalization
    - Windowed dataset generation

Intended for use with forecasting models in the Electronic Trading App.
"""

# Standard library imports

# Third-party imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Local application imports
from feature_engineer import FeatureEngineer
from data_splitter import DataSplitter
from window_generator import WindowGenerator


class DataProcessor:
    """Orchestrates feature engineering, splitting, normalization, and windowing.

    Attributes:
        raw_df (pd.DataFrame): The raw input dataframe.
        feature_set (list[dict]): Features to generate or include.
        forecast_period (int): Prediction horizon in timesteps.
        normalise (bool): Whether to normalize numeric features.
        train_ratio (float): Proportion of data to allocate to training.
        val_ratio (float): Proportion of data to allocate to validation.
        df (pd.DataFrame): Processed dataframe after feature engineering.
        train_df (pd.DataFrame): Training split dataframe.
        val_df (pd.DataFrame): Validation split dataframe.
        test_df (pd.DataFrame): Test split dataframe.
        normalizer (Normalizer | None): Fitted normalizer/scaler.
        window (WindowGenerator): Windowed datasets for ML models.
    """


    def __init__(self, raw_data, feature_set, forecast_period=1, input_width=1, normalise=True, train_ratio=0.8, val_ratio=0.1):
        """Initializes the data processing pipeline.

        Args:
            raw_data (pd.DataFrame): Raw input data.
            feature_set (list[dict]): Feature definitions or column names.
            forecast_period (int, optional): Prediction horizon in timesteps.
                Defaults to 1.
            normalise (bool, optional): Whether to normalize numeric features.
                Defaults to True.
            train_ratio (float, optional): Proportion of data for training.
                Defaults to 0.8.
            val_ratio (float, optional): Proportion of data for validation.
                Defaults to 0.1.
        """
        print("Creating DataProcessor.")

        self.raw_df = pd.DataFrame(raw_data)
        self.feature_set_def = feature_set
        self.forecast_period = forecast_period
        self.input_width = input_width
        self.normalise = normalise
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # Step 1: Feature engineering
        fe = FeatureEngineer(self.raw_df, feature_set)
        self.df = fe.apply().dropna()

        # Explicitly enforce target creation
        target_col = self.get_target()
        print(target_col)
        print(self.df.columns)
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' missing after feature engineering.")

        # Filter: keep only declared features + target
        feature_names = [f["name"] for f in self.feature_set_def.get("features", [])]
        keep_cols = feature_names
        # keep_cols = feature_names + [target_col]
        self.df = self.df[keep_cols]

        print("Final engineered columns:", self.df.columns.tolist())

        # Filter to only keep columns that were engineered
        # feature_set here is a list of column names like ["price", "volume", "target"]
        # if isinstance(self.feature_set, list) and all(isinstance(f, str) for f in self.feature_set):
        #     # feature_set is list of column names
        #     self.df = self.df[self.feature_set]
        # else:
        #     # If feature_set is in dict format, extract the column names
        #     # This shouldn't happen with your current setup, but handle it anyway
        #     pass  # Let FeatureEngineer handle it

        # # Filter to only keep columns specified in feature_set
        # feature_columns = [f for f in self.feature_set if f != "target"]
        # if "target" not in feature_columns:
        #     feature_columns.append("target")
        # self.df = self.df[feature_columns]

        print(self.df.head(5))
        print("Features engineered.")

        # Step 2: Splitting
        splitter = DataSplitter(self.df, train_ratio, val_ratio)
        self.train_df, self.val_df, self.test_df = splitter.split()

        print(self.train_df.head(5))
        print("Data split.")

        # Step 3: Normalisation (optional)
        self.scaler = None
        if self.normalise:
            numeric_cols = self.train_df.select_dtypes(include=["number"]).columns
            self.numeric_cols = numeric_cols

            self.scaler = StandardScaler()
            self.scaler.fit(self.train_df[numeric_cols])

            self.train_df.loc[:, numeric_cols] = self.scaler.transform(self.train_df[numeric_cols])
            self.val_df.loc[:, numeric_cols]   = self.scaler.transform(self.val_df[numeric_cols])
            self.test_df.loc[:, numeric_cols]  = self.scaler.transform(self.test_df[numeric_cols])
            print("Data normalised.")        

        print(self.train_df.head(5))
        
        # Step 4: Windowing (using forecastPeriod)
        self.window = WindowGenerator(
            input_width=self.input_width,
            label_width=1,
            shift=self.forecast_period,
            train_df=self.train_df,
            val_df=self.val_df,
            test_df=self.test_df,
            label_columns=["target"]
        )

        print("Data processed.")
        
        example_inputs, example_labels = next(iter(self.window.train))
        print("inputs:", example_inputs.shape)
        print("labels:", example_labels.shape)
        print(example_inputs[0, :, 0])
        print(example_labels[0, :, 0])


    def get_data(self):
        """
        Return processed datasets in a format usable by ModelHandler.

        Returns:
            dict: {
                'train': windowed training dataset,
                'val': windowed validation dataset,
                'test': windowed test dataset,
                'x_train': pd.DataFrame of training features,
                'y_train': pd.Series of training targets,
                'x_val': pd.DataFrame of validation features,
                'y_val': pd.Series of validation targets,
                'x_test': pd.DataFrame of test features,
                'y_test': pd.Series of test targets
            }
        """
        # Keep numeric columns only
        # self.train_df = self.train_df.select_dtypes(include=["number"])
        # self.val_df   = self.val_df.select_dtypes(include=["number"])
        # self.test_df  = self.test_df.select_dtypes(include=["number"])

        # Split features and target
        target_col = self.get_target()
        x_train, y_train = self.train_df.drop(columns=[target_col]), self.train_df[target_col]
        x_val,   y_val   = self.val_df.drop(columns=[target_col]),   self.val_df[target_col]
        x_test,  y_test  = self.test_df.drop(columns=[target_col]),  self.test_df[target_col]

        # Rebuild window generator with clean data
        self.window = WindowGenerator(
            input_width=self.input_width,
            label_width=self.forecast_period,
            shift=self.forecast_period,
            train_df=self.train_df,
            val_df=self.val_df,
            test_df=self.test_df,
            label_columns=[target_col]
        )

        return {
            "train": self.window.train,
            "val": self.window.val,
            "test": self.window.test,
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test
        }

    def get_window(self):
        """Return window.

        Returns:
            WindowGenerator.
        """   
        return self.window
        
    def get_target(self):
        """Return the target column name."""
        # Case 1: explicit top-level target (preferred format)
        if "target" in self.feature_set_def:
            return "target"

        # Case 2: target embedded in features list
        for f in self.feature_set_def.get("features", []):
            if f["name"] == "target":
                return "target"

        raise ValueError("No target specified in feature set definition.")    
    # def get_target(self):
    #     """Return the name of the target column as defined in the feature set config."""
    #     if isinstance(self.feature_set, dict) and "target" in self.feature_set:
    #         # Explicitly defined in JSON
    #         target_fields = self.feature_set["target"].get("input_data_fields", [])
    #         if target_fields:
    #             # Use first field or enforce one target
    #             return self.feature_set["target"].get("name", target_fields[0])
    #         return self.feature_set["target"].get("name", "target")
    #     else:
    #         raise ValueError("No target specified in feature set definition.")
    
    def get_normalise(self):
        """Return normalise indicator.

        Returns:
            bool: indicator of whether to normalise numeric columns.
        """   
        return self.normalise
    
    def get_normalisation_params(self, col):
        """Return normalisation parameters.

        Args:
            col: the column for which the normalisation parameters are required.

        Returns:
            dictionary: { mean, std } for the given column.
        """   
        if not self.scaler or col not in self.numeric_cols:
            return None
        idx = list(self.numeric_cols).index(col)
        return {
            "mean": self.scaler.mean_[idx],
            "std": self.scaler.scale_[idx]
        }

    def inverse_transform(self, arr, col):
        """Return inverse transform of given time series.

        Args:
            arr (np.array ?): time series to be de-normalised.
            col (int): column of time series, so that normalisation parameters can be retrieved.

        Returns:
            np array ?: de-normalised time series.
        """   
        if not self.scaler or col not in self.numeric_cols:
            return arr
        idx = list(self.numeric_cols).index(col)
        mean, std = self.scaler.mean_[idx], self.scaler.scale_[idx]
        return (arr * std) + mean
