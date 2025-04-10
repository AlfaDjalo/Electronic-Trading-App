import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from stock_data import StockData, DEFAULT_START_DATE, DEFAULT_END_DATE
from ml_data import MLData

@pytest.fixture
def sample_stock_data():
    """Fixture to create sample stock data."""
    dates = pd.date_range(start="2022-01-01", end="2022-01-10", freq="D")
    data = {
        "open": np.random.rand(len(dates)),
        "high": np.random.rand(len(dates)),
        "low": np.random.rand(len(dates)),
        "close": np.random.rand(len(dates)),
        "volume": np.random.randint(100, 1000, len(dates)),
    }
    return pd.DataFrame(data, index=dates)

def test_stock_data_initialization(sample_stock_data, monkeypatch):
    """Test initialization of StockData."""
    def mock_load_data(self):
        self.data = sample_stock_data

    monkeypatch.setattr(StockData, "load_data", mock_load_data)
    stock_data = StockData(ticker="AAPL", load_data=True)
    assert not stock_data.data.empty
    assert "close" in stock_data.data.columns

def test_stock_data_load_lob_data(monkeypatch, tmp_path):
    """Test loading LOB data in StockData."""
    lob_data = pd.DataFrame({
        "timestamp": pd.date_range(start="2022-01-01", periods=10, freq="T").astype(int) // 10**9,
        "lob_feature": np.random.rand(10),
    })
    lob_file = tmp_path / "lob_data.csv"
    lob_data.to_csv(lob_file, index=False)

    stock_data = StockData(ticker="AAPL", use_lob_data=True, lob_filepath=str(lob_file))
    assert stock_data.load_lob_data(str(lob_file))
    assert not stock_data.lob_data.empty
    assert "lob_feature" in stock_data.lob_data.columns

def test_ml_data_initialization(sample_stock_data):
    """Test initialization of MLData."""
    ml_data = MLData(
        raw_data=sample_stock_data,
        lag_period=3,
        forecast_period=1,
        feature_column="close",
        log_returns=False,
        standardised=False,
    )
    assert ml_data.get_data()["x_train"] is not None
    assert ml_data.get_data()["y_train"] is not None

def test_ml_data_lagged_features(sample_stock_data):
    """Test creation of lagged features in MLData."""
    ml_data = MLData(
        raw_data=sample_stock_data,
        lag_period=3,
        forecast_period=1,
        feature_column="close",
    )
    assert any(col.startswith("min_") for col in ml_data.get_features())

def test_ml_data_standardisation(sample_stock_data):
    """Test standardisation of features in MLData."""
    ml_data = MLData(
        raw_data=sample_stock_data,
        lag_period=3,
        forecast_period=1,
        feature_column="close",
        standardised=True,
    )
    x_train = ml_data.get_data()["x_train"]
    assert np.isclose(x_train.mean().mean(), 0, atol=1e-1)
    assert np.isclose(x_train.std().mean(), 1, atol=1e-1)
