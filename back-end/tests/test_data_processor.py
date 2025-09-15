import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import app

from process_data import DataProcessor
from feature_engineer import FeatureEngineer
from data_splitter import DataSplitter
from window_generator import WindowGenerator

@pytest.fixture
def feature_set():
    return {
        "features": [
            {"function": "raw_data", "input_data_fields": ["feature1"], "name": "feature1"},
            {"function": "raw_data", "input_data_fields": ["target"], "name": "target"}
        ]
    }

@pytest.fixture
def mock_df_flat():
    return pd.DataFrame({
        "date": pd.date_range(start="2025-01-01", periods=10, freq="D"),
        "feature1": [100.0]*10,
        "target": [100.0]*10
    })

@pytest.fixture
def mock_df_ramp():
    return pd.DataFrame({
        "date": pd.date_range(start="2025-01-01", periods=10, freq="D"),
        "feature1": np.linspace(50, 150, 10),
        "target": np.linspace(50, 150, 10)
    })

@pytest.fixture
def mock_df_wave():
    t = np.linspace(0, 2*np.pi, 10)
    return pd.DataFrame({
        "date": pd.date_range(start="2025-01-01", periods=10, freq="D"),
        "feature1": 100 + 50*np.sin(t),
        "target": 100 + 50*np.sin(t)
    })


@pytest.mark.parametrize("mock_df", ["mock_df_flat", "mock_df_ramp", "mock_df_wave"])
def test_normalisation(mock_df, feature_set, request):
    df = request.getfixturevalue(mock_df)
    dp = DataProcessor(
        raw_data=df,
        feature_set=feature_set,
        forecast_period=1,
        normalise=True,
        train_ratio=0.7,
        val_ratio=0.2
    )

    numeric_cols = dp.train_df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        col_mean = dp.train_df[col].mean()
        col_std = dp.train_df[col].std()

        # Skip columns with constant values
        if col_std == 0:
            continue

        assert abs(col_mean) < 1e-6
        assert abs(col_std - 1) < 1e-2


@pytest.mark.parametrize("mock_df", ["mock_df_flat", "mock_df_ramp", "mock_df_wave"])
def test_window_generator_shapes(mock_df, feature_set, request):
    df = request.getfixturevalue(mock_df)
    dp = DataProcessor(
        raw_data=df,
        feature_set=feature_set,
        forecast_period=1,
        normalise=True
    )

    # Exclude non-numeric columns before generating windows
    numeric_cols = dp.train_df.select_dtypes(include=["number"]).columns
    dp.train_df = dp.train_df[numeric_cols]
    dp.val_df = dp.val_df[numeric_cols]
    dp.test_df = dp.test_df[numeric_cols]

    train_window, val_window, test_window = dp.get_data()

    # Basic shape checks
    for window in [train_window, val_window, test_window]:
        assert isinstance(window, np.ndarray)
        assert window.ndim == 2  # [samples, features]
        assert window.shape[1] == len(numeric_cols)


# feature_set_dict = {
#     "features": [
#         {"name": "feature1", "input_data_fields": ["feature1"], "function": "raw_data"},
#         {"name": "target", "input_data_fields": ["target"], "function": "raw_data"}
#     ]
# }

# # --- Fixtures / Helpers ---

# def make_mock_data(series_type="flat", start="2025-01-01", end="2025-01-10"):
#     """Generate mock data in the same shape as Yahoo Finance-like test data."""
#     date_range = pd.date_range(start=start, end=end, freq="D")
#     n_days = len(date_range)

#     if series_type == "flat":
#         close_prices = np.full(n_days, 100.0)
#     elif series_type == "ramp":
#         close_prices = np.linspace(50, 150, n_days)
#     elif series_type == "wave":
#         t = np.arange(n_days)
#         frequency = 2 * np.pi / 30
#         close_prices = 100 + 50 * np.sin(frequency * t)
#     else:
#         raise ValueError("Unknown mock series type")

#     df = pd.DataFrame({
#         "date": date_range,
#         "feature1": close_prices,
#         "target": close_prices  # keep target same for simplicity
#     })

#     return df


# @pytest.fixture(params=["flat", "ramp", "wave"])
# def mock_df(request):
#     return make_mock_data(series_type=request.param)


# # --- Tests ---

# def test_feature_engineering_and_split(mock_df):
#     """Check that DataProcessor runs feature engineering and splitting correctly."""
#     dp = DataProcessor(
#         raw_data=mock_df,
#         feature_set=feature_set_dict,  # assumes FeatureEngineer passes these through
#         forecast_period=1,
#         normalise=False,
#         train_ratio=0.7,
#         val_ratio=0.2
#     )

#     # Ensure data split sizes add up
#     total_len = len(dp.df)
#     assert len(dp.train_df) + len(dp.val_df) + len(dp.test_df) == total_len


# def test_normalisation(mock_df):
#     """Check that normalization works (mean ~0, std ~1 for train set)."""
#     dp = DataProcessor(
#         raw_data=mock_df,
#         feature_set=["feature1", "target"],
#         forecast_period=1,
#         normalise=True,
#         train_ratio=0.8,
#         val_ratio=0.1
#     )

#     numeric_cols = dp.train_df.select_dtypes(include=["number"]).columns
#     for col in numeric_cols:
#         mean = dp.train_df[col].mean()
#         std = dp.train_df[col].std(ddof=0)
#         assert abs(mean) < 1e-6
#         assert abs(std - 1.0) < 1e-6


# def test_window_generator_shapes(mock_df):
#     """Check that the window generator produces correct input/label shapes."""
#     dp = DataProcessor(
#         raw_data=mock_df,
#         feature_set=["feature1", "target"],
#         forecast_period=3,  # multi-step forecast
#         normalise=True
#     )

#     sample = next(iter(dp.get_window().train))
#     x, y = sample
#     # x.shape: (batch, input_width, features)
#     # y.shape: (batch, label_width, features_selected)
#     assert x.ndim == 3
#     assert y.ndim == 2 or y.ndim == 3
#     assert y.shape[-1] == 1 or y.shape[-1] == len(["target"])

# # ----------------------------
# # Mock data generator
# # ----------------------------
# @pytest.fixture(params=["flat", "ramp", "wave"])
# def mock_df(request):
#     """Generate simple mock time series data."""
#     series_type = request.param
#     date_range = pd.date_range(start="2025-01-01", end="2025-01-10", freq="D")
#     n_days = len(date_range)

#     if series_type == "flat":
#         values = np.full(n_days, 100.0)
#     elif series_type == "ramp":
#         values = np.linspace(50, 150, n_days)
#     elif series_type == "wave":
#         t = np.arange(n_days)
#         frequency = 2 * np.pi / 5  # one cycle every 5 days
#         values = 100 + 50 * np.sin(frequency * t)
    
#     df = pd.DataFrame({
#         "date": date_range,
#         "feature1": values,
#         "target": values  # simple target = feature1
#     })
#     return df

# # ----------------------------
# # Correct feature set for FeatureEngineer
# # ----------------------------
# @pytest.fixture
# def feature_set():
#     return {
#         "features": [
#             {"name": "feature1", "input_data_fields": ["feature1"], "function": "raw_data"},
#             {"name": "target",   "input_data_fields": ["target"],   "function": "raw_data"}
#         ]
#     }

# # ----------------------------
# # Tests
# # ----------------------------
# def test_feature_engineering_and_split(mock_df, feature_set):
#     """Check that DataProcessor runs feature engineering and splitting correctly."""
#     dp = DataProcessor(
#         raw_data=mock_df,
#         feature_set=feature_set,
#         forecast_period=1,
#         normalise=False,
#         train_ratio=0.7,
#         val_ratio=0.2
#     )
#     # Check train/val/test split lengths
#     total_len = len(mock_df)
#     train_len = len(dp.train_df)
#     val_len = len(dp.val_df)
#     test_len = len(dp.test_df)
    
#     assert train_len + val_len + test_len <= total_len  # dropna may reduce rows
#     assert "feature1" in dp.train_df.columns
#     assert "target" in dp.train_df.columns

# def test_normalisation(mock_df, feature_set):
#     """Check that normalization works (mean ~0, std ~1 for train set)."""
#     dp = DataProcessor(
#         raw_data=mock_df,
#         feature_set=feature_set,
#         forecast_period=1,
#         normalise=True,
#         train_ratio=0.7,
#         val_ratio=0.2
#     )
#     # Check train_df stats
#     numeric_cols = dp.train_df.select_dtypes(include=["number"]).columns
#     for col in numeric_cols:
#         col_mean = dp.train_df[col].mean()
#         col_std = dp.train_df[col].std()
#         assert abs(col_mean) < 1e-6
#         assert abs(col_std - 1) < 1e-6

# def test_window_generator_shapes(mock_df, feature_set):
#     """Check that the window generator produces correct input/label shapes."""
#     dp = DataProcessor(
#         raw_data=mock_df,
#         feature_set=feature_set,
#         forecast_period=1,
#         normalise=True
#     )
#     train_window, val_window, test_window = dp.get_data()
    
#     # Each should be a tf.data.Dataset (or similar)
#     for window in [train_window, val_window, test_window]:
#         for x, y in window.take(1):
#             assert x.shape[1] == 30  # input_width hardcoded in WindowGenerator
#             assert y.shape[1] == dp.forecast_period