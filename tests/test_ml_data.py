import unittest
import pandas as pd
import numpy as np  # Added import for np
from ml_data import MLData

class TestMLData(unittest.TestCase):
    def setUp(self):
        self.raw_data = pd.DataFrame({
            "close": [1, 2, 3, 4, 5],
            "volume": [100, 200, 300, 400, 500]
        })
        self.feature_set = {
            "features": [
                {"name": "close_lagged_1", "input_data_fields": ["close"], "function": "create_lagged_features", "function_parameters": {"num_lags": 1}}
            ],
            "target": {"input_data_fields": ["close"], "function": "create_lagged_features", "function_parameters": {"num_lags": -1}}
        }

    def test_create_lagged_features(self):
        ml_data = MLData(self.raw_data, feature_set_name="Daily data historical")
        lagged = ml_data.create_lagged_features(["close"], num_lags=1)
        self.assertEqual(lagged.iloc[1], 1)
        self.assertTrue(pd.isna(lagged.iloc[0]))

    def test_create_average(self):
        ml_data = MLData(self.raw_data, feature_set_name="Daily data historical")
        avg = ml_data.create_average(["close", "volume"])
        self.assertAlmostEqual(avg.iloc[0], 50.5, places=6)  # Adjusted to use assertAlmostEqual for precision

    def test_apply_feature_set(self):
        ml_data = MLData(self.raw_data, feature_set=self.feature_set)
        ml_data.apply_feature_set(self.feature_set)
        self.assertIn("close_lagged_1", ml_data.data.columns)
        self.assertIn("target", ml_data.data.columns)

    def test_set_feature_set(self):
        ml_data = MLData(self.raw_data)
        ml_data.set_feature_set(feature_set=self.feature_set)
        self.assertEqual(ml_data.feature_set, self.feature_set)

    def test_set_feature_set_name(self):
        ml_data = MLData(self.raw_data)
        ml_data.set_feature_set(feature_set_name="Daily data historical")
        self.assertEqual(ml_data.feature_set_name, "Daily data historical")

    def test_normalise_columns(self):
        """Test normalising columns."""
        raw_data = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
        ml_data = MLData(raw_data, feature_set_name="Daily data historical")
        ml_data.x_train = raw_data
        ml_data.x_test = raw_data
        ml_data.normalise_columns(["close"], ml_data.x_train, ml_data.x_test)
        self.assertAlmostEqual(ml_data.x_train["close"].mean(), 0, places=6)
        self.assertAlmostEqual(ml_data.x_train["close"].std(), 1, places=6)

    def test_calculate_log_returns(self):
        """Test calculating log returns."""
        raw_data = pd.DataFrame({"close": [1, 2, 4, 8, 16]})
        ml_data = MLData(raw_data, feature_set_name="Daily data historical")
        ml_data.calculate_log_returns("close")
        self.assertIn("close_log_returns", ml_data.data.columns)
        self.assertAlmostEqual(ml_data.data["close_log_returns"].iloc[1], np.log(2), places=6)
