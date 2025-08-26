import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np

from stock_data import StockData, DEFAULT_START_DATE, DEFAULT_END_DATE
from ml_data import MLData

def test_stock_data_initialization(sample_stock_data, monkeypatch):
    """Test initialization of StockData."""
    def mock_load_daily_data(self):
        self.data = sample_stock_data

    monkeypatch.setattr(StockData, "load_daily_data", mock_load_daily_data)
    stock_data = StockData(data_type="daily", ticker="ANZ.AX", load_data=True)
    assert not stock_data.data.empty
    assert "close" in stock_data.data.columns

def test_stock_data_load_lob_data(monkeypatch, tmp_path):
    """Test loading LOB data in StockData."""
    lob_data = pd.DataFrame({
        "timestamp": pd.date_range(start="2022-01-01", periods=10, freq="min").astype(int) // 10**9,
        "bid_price_0": np.random.rand(10),
        "ask_price_0": np.random.rand(10),
    })
    lob_file = tmp_path / "lob_data.csv"
    lob_data.to_csv(lob_file, index=False)

    stock_data = StockData(data_type="intraday", ticker="ANZ.AX", load_data=False)
    assert stock_data.load_lob_data(str(lob_file))
    assert not stock_data.data.empty
    assert "mid_price" in stock_data.data.columns

def test_ml_data_initialization(sample_stock_data, sample_feature_set):
    """Test initialization of MLData."""
    ml_data = MLData(
        raw_data=sample_stock_data,
        train_percentage=0.8,
        feature_set=sample_feature_set,
        normalise=False
    )
    data = ml_data.get_data()
    assert data["x_train"] is not None
    assert data["y_train"] is not None

def test_ml_data_apply_feature_set(sample_stock_data, sample_feature_set):
    """Test creation of lagged features in MLData."""
    ml_data = MLData(
        raw_data=sample_stock_data,
        train_percentage=0.8,
        feature_set=sample_feature_set,
        normalise=False
    )
    assert "target" in ml_data.data.columns
    assert "close" in ml_data.data.columns

def test_ml_data_lagged_features(sample_stock_data, sample_feature_set):
    """Test creation of lagged features in MLData."""
    ml_data = MLData(
        raw_data=sample_stock_data,
        train_percentage=0.8,
        feature_set=sample_feature_set,
        normalise=False
    )
    lagged_feature = ml_data.create_lagged_features(input_field=["close"], num_lags=3)
    # assert lagged_feature is not None
    # assert lagged_feature.isna().sum() > 0  # Lagged features should have NaN values at the start
    assert lagged_feature.iloc[3]==1
    assert pd.isna(lagged_feature.iloc[0])

def test_ml_data_normalisation(sample_stock_data, sample_feature_set):
    """Test normalisation of features in MLData."""
    ml_data = MLData(
        raw_data=sample_stock_data,
        train_percentage=0.8,
        feature_set=sample_feature_set,
        normalise=True
    )
    x_train = ml_data.get_data()["x_train"]
    assert np.isclose(x_train.mean().mean(), 0, atol=1e-1)
    assert np.isclose(x_train.std().mean(), 1, atol=1e-1)

def test_create_average(sample_stock_data, sample_feature_set):
    """Test create average of features in MLData."""
    ml_data = MLData(
        raw_data=sample_stock_data,
        train_percentage=0.8,
        feature_set=sample_feature_set,
        normalise=True
    )

    avg = ml_data.create_average(["close", "volume"])
    assert np.isclose(avg.iloc[0], 50.5, atol=1e-1)

def test_calculate_log_returns(sample_stock_data, sample_feature_set):
    """Test calculating log returns."""
    ml_data = MLData(
        raw_data=sample_stock_data,
        train_percentage=0.8,
        feature_set=sample_feature_set,
        normalise=True
    )
    ml_data.calculate_log_returns("close")
    assert "close_log_returns" in ml_data.data.columns
    assert np.isclose(ml_data.data["close_log_returns"].iloc[0], np.log(2), atol=1e-1)


