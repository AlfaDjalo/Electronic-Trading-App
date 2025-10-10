import pytest
import sys
import os
import numpy as np
import pandas as pd
import tempfile
import json

# Add parent directory to path (following your pattern)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.model_service import process_models_request

# =============================================================================
# REALISTIC TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def realistic_time_series_data():
    """Create realistic time series data similar to stock/financial data."""
    np.random.seed(42)  # For reproducible tests
    
    # Create 6 months of daily data
    dates = pd.date_range(start="2023-01-01", end="2023-06-30", freq="D")
    n_points = len(dates)
    
    # Generate realistic time series with multiple components
    t = np.arange(n_points)
    
    # Base price with trend
    base_price = 100.0
    trend = 0.02 * t  # Gradual upward trend
    
    # Seasonal patterns (weekly cycle)
    weekly_pattern = 5 * np.sin(2 * np.pi * t / 7)
    
    # Market volatility (random walk component)
    volatility = np.cumsum(np.random.normal(0, 2, n_points))
    
    # Combine components
    price = base_price + trend + weekly_pattern + volatility
    
    # Create correlated features
    volume = 1000000 + 50000 * np.random.randn(n_points) + 10000 * np.abs(np.diff(np.concatenate([[price[0]], price])))
    
    # Technical indicators
    # Simple moving averages (simplified)
    ma_5 = pd.Series(price).rolling(window=5, min_periods=5).mean().values
    ma_20 = pd.Series(price).rolling(window=20, min_periods=20).mean().values
    
    # Volatility measure
    returns = np.diff(np.concatenate([[price[0]], price])) / price * 100
    volatility_measure = pd.Series(returns).rolling(window=10, min_periods=1).std().fillna(0).values
    
    return pd.DataFrame({
        'date': dates,
        'price': price,
        'volume': volume,
        'ma_5': ma_5,
        'ma_20': ma_20,
        'volatility': volatility_measure,
        'target': price  # Target is the price we want to predict
    })

@pytest.fixture
def mock_feature_sets_file():
    """Create temporary feature sets file."""
    feature_sets_config = {
        "basic": {
            "features": [
                {"function": "raw_data", "input_data_fields": ["price"], "name": "price"},
                {"function": "raw_data", "input_data_fields": ["target"], "name": "target"}
            ]
        },
        "advanced": {
            "features": [
                {"function": "raw_data", "input_data_fields": ["price"], "name": "price"},
                {"function": "raw_data", "input_data_fields": ["volume"], "name": "volume"},
                {"function": "raw_data", "input_data_fields": ["ma_5"], "name": "ma_5"},
                {"function": "raw_data", "input_data_fields": ["ma_20"], "name": "ma_20"},
                {"function": "raw_data", "input_data_fields": ["target"], "name": "target"}
            ]
        },
        "minimal": {
            "features": [
                {"function": "raw_data", "input_data_fields": ["target"], "name": "target"}
            ]
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(feature_sets_config, f)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    try:
        os.unlink(temp_file)
    except OSError:
        pass

@pytest.fixture
def realistic_hyperparameters():
    """Realistic hyperparameters for system testing."""
    return {
        "train_val_test_split": [0.7, 0.2, 0.1],
        "some_other_param": "test_value"
    }

# =============================================================================
# SYSTEM TESTS
# =============================================================================

class TestSystemWorkflow:
    """End-to-end system tests using the service layer with real components."""
    
    def test_complete_single_model_workflow(self, realistic_time_series_data, 
                                          mock_feature_sets_file, realistic_hyperparameters):
        """Test complete workflow with a single model using real components."""
        
        # Convert DataFrame to API format (list of dicts)
        raw_data_list = realistic_time_series_data.to_dict('records')
        for record in raw_data_list:
            record['date'] = record['date'].strftime('%Y-%m-%d')
        
        model_list = [{
            "name": "system_test_linear",
            "model": "linear",
            "featureSet": "basic",
            "forecastPeriod": 1,
            "inputWidth": 10,
            "normalise": True,
            "params": {
                "epochs": 5,  # Small for testing speed
                "batch_size": 32,
                "optimizer": "adam",
                "loss": "mse"
            }
        }]
        
        # Process request using service layer with real components
        results = process_models_request(
            raw_data_list=raw_data_list,
            model_list=model_list,
            hyperparameters=realistic_hyperparameters,
            feature_sets_file=mock_feature_sets_file,
            verbose=True
        )
        
        # Verify complete result structure
        assert isinstance(results, dict)
        assert results['success'] == True
        assert 'dates' in results
        assert 'actual' in results
        assert 'predictions' in results
        assert 'stats' in results
        assert 'metadata' in results
        
        # Verify dates
        dates = results['dates']
        assert isinstance(dates, list)
        assert len(dates) > 0
        assert all(isinstance(d, str) for d in dates)
        
        # Verify actual values
        actual = results['actual']
        assert isinstance(actual, list)
        assert len(actual) > 0
        assert all(isinstance(v, (int, float)) for v in actual)
        
        # Verify predictions
        model_name = "system_test_linear"
        assert model_name in results['predictions']
        predictions = results['predictions'][model_name]
        assert isinstance(predictions, list)
        assert len(predictions) == len(actual)
        assert all(isinstance(p, (int, float)) for p in predictions)
        
        # Verify stats
        assert model_name in results['stats']
        stats = results['stats'][model_name]
        assert isinstance(stats, dict)
        required_metrics = ['mse', 'mae', 'rmse', 'r2']
        for metric in required_metrics:
            assert metric in stats
            assert isinstance(stats[metric], (int, float))
        
        # Verify metadata
        metadata = results['metadata']
        assert metadata['total_models'] == 1
        assert metadata['successful_models'] == 1
        assert metadata['failed_models'] == 0
    
    def test_complete_multiple_model_workflow(self, realistic_time_series_data,
                                            mock_feature_sets_file, realistic_hyperparameters):
        """Test complete workflow with multiple models."""
        
        raw_data_list = realistic_time_series_data.to_dict('records')
        for record in raw_data_list:
            record['date'] = record['date'].strftime('%Y-%m-%d')
        
        model_list = [
            {
                "name": "system_linear",
                "model": "linear",
                "featureSet": "basic",
                "forecastPeriod": 1,
                "inputWidth": 8,
                "normalise": True,
                "params": {"epochs": 3, "batch_size": 16}
            },
            {
                "name": "system_mlp",
                "model": "mlp",
                "featureSet": "advanced",
                "forecastPeriod": 1,
                "inputWidth": 10,
                "normalise": False,
                "params": {"epochs": 3, "hidden_units": [32, 16]}
            },
            {
                "name": "system_baseline",
                "model": "baseline",
                "featureSet": "minimal",
                "forecastPeriod": 1,
                "inputWidth": 5,
                "normalise": False,
                "params": {"epochs": 1}
            }
        ]
        
        results = process_models_request(
            raw_data_list=raw_data_list,
            model_list=model_list,
            hyperparameters=realistic_hyperparameters,
            feature_sets_file=mock_feature_sets_file,
            verbose=True
        )
        
        # Verify overall structure
        assert isinstance(results, dict)
        assert results['success'] == True
        assert 'dates' in results
        assert 'actual' in results
        assert 'predictions' in results
        assert 'stats' in results
        
        # Verify all models are represented
        expected_models = ["system_linear", "system_mlp", "system_baseline"]
        assert len(results['predictions']) == len(expected_models)
        assert len(results['stats']) == len(expected_models)
        
        for model_name in expected_models:
            assert model_name in results['predictions']
            assert model_name in results['stats']
            
            # Verify predictions structure
            predictions = results['predictions'][model_name]
            assert isinstance(predictions, list)
            assert len(predictions) > 0
            
            # Verify stats structure
            stats = results['stats'][model_name]
            if 'error' not in stats:  # Model succeeded
                assert 'mse' in stats
                assert 'mae' in stats
                assert 'rmse' in stats
                assert 'r2' in stats
    
    def test_normalization_end_to_end(self, realistic_time_series_data,
                                     mock_feature_sets_file, realistic_hyperparameters):
        """Test normalization works end-to-end with real data."""
        
        raw_data_list = realistic_time_series_data.to_dict('records')
        for record in raw_data_list:
            record['date'] = record['date'].strftime('%Y-%m-%d')
        
        # Model with normalization
        normalized_model = [{
            "name": "normalized_test",
            "model": "linear",
            "featureSet": "basic",
            "forecastPeriod": 1,
            "inputWidth": 8,
            "normalise": True,
            "params": {"epochs": 2, "batch_size": 16}
        }]
        
        # Model without normalization
        unnormalized_model = [{
            "name": "unnormalized_test",
            "model": "linear",
            "featureSet": "basic",
            "forecastPeriod": 1,
            "inputWidth": 8,
            "normalise": False,
            "params": {"epochs": 2, "batch_size": 16}
        }]
        
        # Run both models
        norm_result = process_models_request(
            raw_data_list=raw_data_list,
            model_list=normalized_model,
            hyperparameters=realistic_hyperparameters,
            feature_sets_file=mock_feature_sets_file
        )
        
        unnorm_result = process_models_request(
            raw_data_list=raw_data_list,
            model_list=unnormalized_model,
            hyperparameters=realistic_hyperparameters,
            feature_sets_file=mock_feature_sets_file
        )
        
        # Both should succeed
        assert norm_result['success'] == True
        assert unnorm_result['success'] == True
        assert 'error' not in norm_result['stats']['normalized_test']
        assert 'error' not in unnorm_result['stats']['unnormalized_test']
        
        # Predictions should be in similar ranges to original data
        # (normalization should be reversed)
        norm_predictions = norm_result['predictions']['normalized_test']
        unnorm_predictions = unnorm_result['predictions']['unnormalized_test']
        
        # Both should predict values in reasonable range relative to input data
        data_range = realistic_time_series_data['target'].max() - realistic_time_series_data['target'].min()
        data_mean = realistic_time_series_data['target'].mean()
        
        for predictions in [norm_predictions, unnorm_predictions]:
            pred_array = np.array(predictions)
            
            # Predictions should be finite
            assert np.all(np.isfinite(pred_array))
            
            # Predictions should be in a reasonable range
            # (allowing for some model error)
            assert np.abs(np.mean(pred_array) - data_mean) < data_range * 2
    
    def test_different_forecast_periods(self, realistic_time_series_data,
                                      mock_feature_sets_file, realistic_hyperparameters):
        """Test system works with different forecast periods."""
        
        raw_data_list = realistic_time_series_data.to_dict('records')
        for record in raw_data_list:
            record['date'] = record['date'].strftime('%Y-%m-%d')
        
        forecast_periods = [1, 3, 5]
        
        for forecast_period in forecast_periods:
            model_list = [{
                "name": f"forecast_{forecast_period}",
                "model": "linear",
                "featureSet": "basic",
                "forecastPeriod": forecast_period,
                "inputWidth": 10,
                "normalise": False,
                "params": {"epochs": 2}
            }]
            
            result = process_models_request(
                raw_data_list=raw_data_list,
                model_list=model_list,
                hyperparameters=realistic_hyperparameters,
                feature_sets_file=mock_feature_sets_file
            )
            
            # Should succeed
            assert result['success'] == True
            model_name = f"forecast_{forecast_period}"
            assert model_name in result['predictions']
            assert model_name in result['stats']
            
            # Predictions should have correct length
            predictions = result['predictions'][model_name]
            assert isinstance(predictions, list)
            assert len(predictions) > 0
    
    def test_cache_efficiency_in_system(self, realistic_time_series_data,
                                      mock_feature_sets_file, realistic_hyperparameters):
        """Test that caching provides efficiency gains in real system."""
        
        raw_data_list = realistic_time_series_data.to_dict('records')
        for record in raw_data_list:
            record['date'] = record['date'].strftime('%Y-%m-%d')
        
        # Models with same data configuration
        shared_config_models = [
            {
                "name": "cached_linear",
                "model": "linear",
                "featureSet": "basic",
                "forecastPeriod": 1,
                "inputWidth": 10,
                "normalise": True,
                "params": {"epochs": 2}
            },
            {
                "name": "cached_mlp",
                "model": "mlp",
                "featureSet": "basic",     # Same
                "forecastPeriod": 1,       # Same
                "inputWidth": 10,          # Same
                "normalise": True,         # Same
                "params": {"epochs": 2, "hidden_units": [32]}
            }
        ]
        
        # Run models
        results = process_models_request(
            raw_data_list=raw_data_list,
            model_list=shared_config_models,
            hyperparameters=realistic_hyperparameters,
            feature_sets_file=mock_feature_sets_file,
            verbose=True
        )
        
        # Both models should succeed
        assert results['success'] == True
        assert len(results['predictions']) == 2
        assert 'cached_linear' in results['predictions']
        assert 'cached_mlp' in results['predictions']
        
        # Cache should contain the shared configuration
        cache_info = results['metadata']['cache_info']
        assert cache_info['cached_configurations'] == 1
        
        # The shared config should be in cache
        expected_config = ('basic', 1, 10, True)
        assert expected_config in cache_info['cache_keys']
    
    def test_error_recovery_in_system(self, realistic_time_series_data,
                                    mock_feature_sets_file, realistic_hyperparameters):
        """Test system error recovery with real components."""
        
        raw_data_list = realistic_time_series_data.to_dict('records')
        for record in raw_data_list:
            record['date'] = record['date'].strftime('%Y-%m-%d')
        
        # Mix of valid and invalid configurations
        mixed_configs = [
            {
                "name": "valid_model",
                "model": "linear",
                "featureSet": "basic",
                "forecastPeriod": 1,
                "inputWidth": 8,
                "normalise": False,
                "params": {"epochs": 2}
            },
            {
                "name": "invalid_model",
                "model": "nonexistent_model",  # Invalid model type
                "featureSet": "basic",
                "forecastPeriod": 1,
                "inputWidth": 8,
                "normalise": False,
                "params": {"epochs": 2}
            }
        ]
        
        # Run models - should handle errors gracefully
        results = process_models_request(
            raw_data_list=raw_data_list,
            model_list=mixed_configs,
            hyperparameters=realistic_hyperparameters,
            feature_sets_file=mock_feature_sets_file
        )
        
        # Should have results for both models
        assert results['success'] == True
        assert len(results['predictions']) == 2
        assert len(results['stats']) == 2
        
        # Valid model should succeed
        assert 'valid_model' in results['predictions']
        assert 'error' not in results['stats']['valid_model']
        
        # Invalid model should fail gracefully
        assert 'invalid_model' in results['predictions']
        assert results['predictions']['invalid_model'] == []
        assert 'error' in results['stats']['invalid_model']
    
    def test_realistic_data_quality(self, realistic_time_series_data):
        """Test that our realistic test data has expected properties."""
        
        data = realistic_time_series_data
        
        # Should have expected columns
        expected_columns = ['date', 'price', 'volume', 'ma_5', 'ma_20', 'volatility', 'target']
        assert all(col in data.columns for col in expected_columns)
        
        # Should have reasonable amount of data
        assert len(data) > 100  # At least 100 data points
        
        # Dates should be in order
        assert data['date'].is_monotonic_increasing
        
        # Price should have reasonable variance
        price_std = data['price'].std()
        price_mean = data['price'].mean()
        assert price_std > 0  # Should have some variance
        assert price_std < price_mean  # But not too volatile
        
        # Moving averages should be smoother than raw price
        price_volatility = data['price'].std()
        ma5_volatility = data['ma_5'].std()
        ma20_volatility = data['ma_20'].std()
        
        assert ma5_volatility <= price_volatility  # MA5 should be smoother
        assert ma20_volatility <= ma5_volatility   # MA20 should be even smoother
        
        # No missing values in critical columns
        assert not data[['date', 'price', 'target']].isnull().any().any()
    
    def test_input_validation_integration(self, mock_feature_sets_file):
        """Test input validation through the service layer."""
        
        # Test no models provided
        with pytest.raises(ValueError, match="No models provided"):
            process_models_request(
                raw_data_list=[{'date': '2023-01-01', 'price': 100, 'target': 100}],
                model_list=[],
                hyperparameters={},
                feature_sets_file=mock_feature_sets_file
            )
        
        # Test no data provided
        with pytest.raises(ValueError, match="No data provided"):
            process_models_request(
                raw_data_list=[],
                model_list=[{"name": "test", "model": "linear"}],
                hyperparameters={},
                feature_sets_file=mock_feature_sets_file
            )
        
        # Test missing date column
        with pytest.raises(ValueError, match="Data must contain 'date' column"):
            process_models_request(
                raw_data_list=[{'price': 100, 'target': 100}],  # No date
                model_list=[{"name": "test", "model": "linear"}],
                hyperparameters={},
                feature_sets_file=mock_feature_sets_file
            )