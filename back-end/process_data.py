import json
import pandas as pd
from sklearn.model_selection import train_test_split

class ProcessData:
    def __init__(self, data: pd.DataFrame, feature_set: dict = None):
        """
        data: raw time series DataFrame with at least a 'date' column
        feature_set: dict describing features & target (from JSON config)
        """
        if "date" not in data.columns:
            raise ValueError("Input data must contain a 'date' column.")

        self.data = data.copy()
        self.feature_set = feature_set or {}
        self.features = []
        self.target = None
        self.normalisation_params = {}
        self.is_normalised = False

        # Storage for splits
        self.train_df = None
        self.val_df = None
        self.test_df = None

    # ------------------------------------------------
    # Representation
    # ------------------------------------------------
    def __repr__(self):
        return f"<ProcessData rows={len(self.data)}, features={len(self.features)}, target={self.target}>"

    # ------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------
    def create_lag(self, input_field, num_lags=1, **kwargs):
        field = input_field[0]
        return pd.concat(
            [self.data[field].shift(i).rename(f"{field}_lag{i}") for i in range(1, num_lags+1)],
            axis=1
        )

    def create_rsi(self, input_field, window=14, **kwargs):
        field = input_field[0]
        delta = self.data[field].diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window).mean()
        rs = gain / loss
        return pd.Series(100 - (100 / (1 + rs)), name=f"{field}_RSI{window}")

    def calculate_log_returns(self, input_field, **kwargs):
        field = input_field[0]
        return pd.Series(
            pd.Series(self.data[field]).pct_change().apply(lambda x: pd.np.log1p(x)),
            name=f"{field}_logret"
        )

    # ------------------------------------------------
    # Apply feature set
    # ------------------------------------------------
    def apply_feature_set(self):
        """
        Apply features and target definition from feature_set JSON.
        """
        if not self.feature_set:
            raise ValueError("No feature set defined.")

        for feature in self.feature_set.get("features", []):
            method_name = feature["method"]
            input_field = feature["input"]
            method = getattr(self, method_name, None)
            if not method:
                raise RuntimeError(f"Feature method '{method_name}' not implemented")

            new_feature = method(input_field, **feature.get("params", {}))
            if isinstance(new_feature, pd.Series):
                self.data[new_feature.name] = new_feature
                self.features.append(new_feature.name)
            elif isinstance(new_feature, pd.DataFrame):
                for col in new_feature.columns:
                    self.data[col] = new_feature[col]
                    self.features.append(col)

        # Target
        target_conf = self.feature_set.get("target", {})
        if target_conf:
            field = target_conf["input"][0]
            method_name = target_conf["method"]
            method = getattr(self, method_name, None)
            if not method:
                raise RuntimeError(f"Target method '{method_name}' not implemented")

            self.data["target"] = method(target_conf["input"], **target_conf.get("params", {}))
            self.target = "target"

        # Drop rows with NaNs (common after lagging/rolling)
        self.data.dropna(inplace=True)

    # ------------------------------------------------
    # Train/Val/Test split
    # ------------------------------------------------
    def split_data(self, test_size=0.2, val_size=0.1, shuffle=False):
        """
        Split data into train/val/test.
        """
        train_df, test_df = train_test_split(self.data, test_size=test_size, shuffle=shuffle)

        # Further split validation from train
        train_df, val_df = train_test_split(train_df, test_size=val_size, shuffle=shuffle)

        self.train_df, self.val_df, self.test_df = train_df, val_df, test_df

    # ------------------------------------------------
    # Normalisation
    # ------------------------------------------------
    def normalise(self):
        """
        Normalise features and target based on train statistics.
        """
        if self.is_normalised:
            return

        if self.train_df is None:
            raise RuntimeError("Must call split_data() before normalisation.")

        mean = self.train_df[self.features].mean()
        std = self.train_df[self.features].std().replace(0, 1)

        for df in [self.train_df, self.val_df, self.test_df]:
            df[self.features] = (df[self.features] - mean) / std

        self.normalisation_params = {"mean": mean.to_dict(), "std": std.to_dict()}
        self.is_normalised = True

    # ------------------------------------------------
    # Prepare for API response
    # ------------------------------------------------
    def to_dict(self, max_rows=500):
        """
        Convert processed data into JSON-serializable dict for API.
        """
        preview = self.data.head(max_rows).to_dict(orient="records")

        return {
            "features": self.features,
            "target": self.target,
            "normalisation_params": self.normalisation_params,
            "preview": preview,
            "num_rows": len(self.data),
        }
