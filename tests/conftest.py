import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_stock_data():
    """Fixture to create sample stock data."""
    dates = pd.date_range(start="2022-01-01", end="2022-01-10", freq="D")

    # Get the length of the dates array
    n_days = len(dates)

    # Create ascending integer series for open, high, low, close
    open_series = np.arange(1, n_days+1)
    high_series = np.arange(1, n_days+1)
    low_series = np.arange(1, n_days+1)
    close_series = np.arange(1, n_days+1)

    # Create ascending volume series in multiples of 100
    volume_series = np.arange(100, (n_days+1) * 100, 100)

    # Create the data dictionary
    data = {
        "open": open_series,
        "high": high_series,
        "low": low_series,
        "close": close_series,
        "volume": volume_series,
    }

    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_feature_set():
    """Fixture to create sample stock data."""
    test_feature_set = {
        "features": [
            {
                "name": "Close",
                "input_data_fields": [
                    "close"
                ],
                "function": "raw_data",
                "function_parameters": None
            },
        ],
        "data_type": "daily",
        "target": {
            "input_data_fields": [
                "close"
            ],
            "function": "create_lagged_features",
            "function_parameters": {
                "num_lags": -1
            }
        }
    }
    return test_feature_set

@pytest.fixture
def sample_params():
    """Fixture to create sample model parameters."""
    test_params = {
        "TestModel": {
            "num_units": {
                "type": "integer",
                "default": 10,
                "min": 1,
                "max": 100
            }
        }
    }
    return test_params

