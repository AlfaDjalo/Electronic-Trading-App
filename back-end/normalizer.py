import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Normalizer:
    # def __init__(self):
    #     self.params = {}
    
    # def fit(self, df):
    #     self.params = {col: (df[col].mean(), df[col].std()) for col in df.columns}
    
    # def transform(self, df):
    #     return (df - {col: m for col, (m, _) in self.params.items()}) / {
    #         col: s for col, (_, s) in self.params.items()
    #     }
    
    # def inverse_transform(self, df):
    #     return (df * {col: s for col, (_, s) in self.params.items()}) + {
    #         col: m for col, (m, _) in self.params.items()
    #     }


# class DataProcessor:
    def __init__(self, train_df, val_df, test_df, target_col="target", normalise=True, scaler_type="standard"):
        self.train_df = train_df.copy()
        self.val_df = val_df.copy()
        self.test_df = test_df.copy()
        self.target_col = target_col
        self.normalise = normalise

        if scaler_type == "standard":
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")

    def fit_transform(self):
        """
        Fit scalers on train data only, then transform train/val/test.
        """
        if not self.normalise:
            return self.train_df, self.val_df, self.test_df

        # Separate features/target
        X_train = self.train_df.drop(columns=[self.target_col])
        y_train = self.train_df[[self.target_col]]

        X_val = self.val_df.drop(columns=[self.target_col])
        y_val = self.val_df[[self.target_col]]

        X_test = self.test_df.drop(columns=[self.target_col])
        y_test = self.test_df[[self.target_col]]

        # === Fit only on TRAIN ===
        self.feature_scaler.fit(X_train)
        self.target_scaler.fit(y_train)

        # === Transform ===
        self.train_df = pd.DataFrame(
            self.feature_scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        self.train_df[self.target_col] = self.target_scaler.transform(y_train)

        self.val_df = pd.DataFrame(
            self.feature_scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        self.val_df[self.target_col] = self.target_scaler.transform(y_val)

        self.test_df = pd.DataFrame(
            self.feature_scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        self.test_df[self.target_col] = self.target_scaler.transform(y_test)

        print("✅ Data normalized.")
        return self.train_df, self.val_df, self.test_df

    def save_scalers(self, feature_path="feature_scaler.pkl", target_path="target_scaler.pkl"):
        joblib.dump(self.feature_scaler, feature_path)
        joblib.dump(self.target_scaler, target_path)
        print(f"Scalers saved: {feature_path}, {target_path}")

    def load_scalers(self, feature_path="feature_scaler.pkl", target_path="target_scaler.pkl"):
        self.feature_scaler = joblib.load(feature_path)
        self.target_scaler = joblib.load(target_path)
        print("Scalers loaded.")

    def inverse_transform_target(self, y_scaled):
        """
        Convert model predictions back to original scale.
        """
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
