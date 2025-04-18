import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
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
    def mock_load_daily_data(self):
        self.data = sample_stock_data

    monkeypatch.setattr(StockData, "load_daily_data", mock_load_daily_data)
    stock_data = StockData(data_type="daily", ticker="AAPL", load_data=True)
    assert not stock_data.data.empty
    assert "close" in stock_data.data.columns

def test_stock_data_load_lob_data(monkeypatch, tmp_path):
    """Test loading LOB data in StockData."""
    lob_data = pd.DataFrame({
        "timestamp": pd.date_range(start="2022-01-01", periods=10, freq="T").astype(int) // 10**9,
        "bid_price_0": np.random.rand(10),
        "ask_price_0": np.random.rand(10),
    })
    lob_file = tmp_path / "lob_data.csv"
    lob_data.to_csv(lob_file, index=False)

    stock_data = StockData(data_type="intraday", ticker="AAPL", load_data=False)
    assert stock_data.load_lob_data(str(lob_file))
    assert not stock_data.data.empty
    assert "mid_price" in stock_data.data.columns

def test_ml_data_initialization(sample_stock_data):
    """Test initialization of MLData."""
    ml_data = MLData(
        raw_data=sample_stock_data,
        train_percentage=0.8,
        feature_set_name="Daily data historical",
        normalise=False
    )
    data = ml_data.get_data()
    assert data["x_train"] is not None
    assert data["y_train"] is not None

def test_ml_data_lagged_features(sample_stock_data):
    """Test creation of lagged features in MLData."""
    ml_data = MLData(
        raw_data=sample_stock_data,
        train_percentage=0.8,
        feature_set_name="Daily data historical",
        normalise=False
    )
    lagged_feature = ml_data.create_lagged_features(input_field=["close"], num_lags=3)
    assert lagged_feature is not None
    assert lagged_feature.isna().sum() > 0  # Lagged features should have NaN values at the start

def test_ml_data_normalisation(sample_stock_data):
    """Test normalisation of features in MLData."""
    ml_data = MLData(
        raw_data=sample_stock_data,
        train_percentage=0.8,
        feature_set_name="Daily data historical",
        normalise=True
    )
    x_train = ml_data.get_data()["x_train"]
    assert np.isclose(x_train.mean().mean(), 0, atol=1e-1)
    assert np.isclose(x_train.std().mean(), 1, atol=1e-1)
