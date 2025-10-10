import pytest
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add parent directory to path (following your pattern)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.base import ModelConfig
from training.runner import ModelRunner
from process_data import DataProcessor

# =============================================================================
# TEST FIXTURES (following your pattern)
# =============================================================================

@pytest.fixture
def mock_raw_data():
    """Mock raw time series data similar to your existing fixtures."""
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    
    # Create realistic time series with trend and seasonality
    t = np.arange(200)
    trend = 0.1 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 30)  # 30-day cycle
    noise = np.random.normal(0, 2, 200)
    
    values = 100 + trend + seasonal + noise
    
    return pd.DataFrame({
        "date": dates,
        "value": values,
        "feature1": values + np.random.normal(0, 5, 200),
        "feature2": np.random.normal(50, 10, 200),
        "target": values + np.random.normal(0, 1, 200)
    })

@pytest.fixture
def mock_feature_set_manager():
    """Mock feature set manager."""
    manager = Mock()
    
    # Define different feature sets
    feature_sets = {
        "basic": ["feature1", "target"],
        "advanced": ["feature1", "feature2", "target"],
        "minimal": ["target"]
    }
    
    def get_feature_set(name):
        return feature_sets.get(name, ["feature1", "target"])
    
    manager.get_feature_set.side_effect = get_feature_set
    return manager

@pytest.fixture
def mock_data_processor():
    """Mock DataProcessor that returns realistic processed data."""
    def create_mock_processor(raw_data, feature_set, forecast_period, input_width, normalise, train_ratio, val_ratio):
        processor = Mock()
        
        # Calculate data sizes
        total_size = len(raw_data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # Mock dimensions based on parameters
        num_features = len(feature_set)
        batch_size_train = max(10, train_size // 10)  # Reasonable batch size
        batch_size_val = max(5, val_size // 10)
        batch_size_test = max(5, (total_size - train_size - val_size) // 10)
        
        # Create mock data with proper shapes
        processed_data = {
            'x_train': np.random.randn(batch_size_train, input_width, num_features),
            'y_train': np.random.randn(batch_size_train, forecast_period),
            'x_val': np.random.randn(batch_size_val, input_width, num_features),
            'y_val': np.random.randn(batch_size_val, forecast_period),
            'x_test': np.random.randn(batch_size_test, input_width, num_features),
            'y_test': np.random.randn(batch_size_test, forecast_period)
        }
        
        processor.get_data.return_value = processed_data
        processor.get_normalise.return_value = normalise
        processor.get_target.return_value = "target"
        processor.get_normalisation_params.return_value = {
            "mean": 100.0, "std": 20.0
        } if normalise else None
        
        return processor
    
    return create_mock_processor

@pytest.fixture
def sample_hyperparameters():
    """Sample hyperparameters for testing."""
    return {
        "train_val_test_split": [0.7, 0.15, 0.15],
        "some_other_param": "value"
    }

@pytest.fixture
def single_model_list():
    """Single model configuration for testing."""
    return [{
        "name": "test_linear",
        "model": "linear",
        "featureSet": "basic",
        "forecastPeriod": 1,
        "inputWidth": 10,
        "normalise": False,
        "params": {
            "epochs": 3,
            "batch_size": 16,
            "optimizer": "adam"
        }
    }]

@pytest.fixture
def multiple_model_list():
    """Multiple model configurations for testing."""
    return [
        {
            "name": "linear_model",
            "model": "linear",
            "featureSet": "basic",
            "forecastPeriod": 1,
            "inputWidth": 10,
            "normalise": False,
            "params": {"epochs": 2, "batch_size": 16}
        },
        {
            "name": "lstm_model",
            "model": "lstm",
            "featureSet": "advanced",
            "forecastPeriod": 1,
            "inputWidth": 15,
            "normalise": True,
            "params": {"epochs": 3, "lstm_units": 32}
        },
        {
            "name": "mlp_model",
            "model": "mlp",
            "featureSet": "basic",
            "forecastPeriod": 2,
            "inputWidth": 8,
            "normalise": False,
            "params": {"epochs": 2, "hidden_units": [64, 32]}
        }
    ]

@pytest.fixture
def models_with_same_data_config():
    """Multiple models that share the same data configuration."""
    return [
        {
            "name": "linear_shared",
            "model": "linear",
            "featureSet": "basic",
            "forecastPeriod": 1,
            "inputWidth": 10,
            "normalise": False,
            "params": {"epochs": 2}
        },
        {
            "name": "mlp_shared",
            "model": "mlp",
            "featureSet": "basic",  # Same as above
            "forecastPeriod": 1,    # Same as above
            "inputWidth": 10,       # Same as above
            "normalise": False,     # Same as above
            "params": {"epochs": 2, "hidden_units": [32]}
        }
    ]

# =============================================================================
# TEST CLASS
# =============================================================================

class TestModelRunner:
    """Integration tests for ModelRunner class."""
    
    def test_runner_instantiation(self, mock_raw_data, mock_feature_set_manager, sample_hyperparameters):
        """Test ModelRunner can be instantiated."""
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters,
            verbose=False
        )
        
        assert runner.raw_data is mock_raw_data
        assert runner.feature_set_manager is mock_feature_set_manager
        assert runner.hyperparameters == sample_hyperparameters
        assert runner.verbose == False
        assert runner.train_ratio == 0.7
        assert runner.val_ratio == 0.15
        assert runner.test_ratio == 0.15
    
    def test_runner_with_verbose(self, mock_raw_data, mock_feature_set_manager, sample_hyperparameters):
        """Test ModelRunner with verbose output."""
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters,
            verbose=True
        )
        assert runner.verbose == True
    
    def test_default_train_val_test_split(self, mock_raw_data, mock_feature_set_manager):
        """Test default train/val/test split when not specified."""
        hyperparameters = {}  # No split specified
        
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=hyperparameters
        )
        
        assert runner.train_ratio == 0.8  # Default
        assert runner.val_ratio == 0.1    # Default
        assert runner.test_ratio == 0.1   # Default
    
    def test_group_models_by_data_config(self, mock_raw_data, mock_feature_set_manager, 
                                       sample_hyperparameters, multiple_model_list):
        """Test that models are correctly grouped by data configuration."""
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        groups = runner._group_models_by_data_config(multiple_model_list)
        
        # Should have 3 groups (all different configurations)
        assert len(groups) == 3
        
        # Check that each group has the expected model
        group_keys = list(groups.keys())
        assert ("basic", 1, 10, False) in group_keys      # linear_model
        assert ("advanced", 1, 15, True) in group_keys    # lstm_model  
        assert ("basic", 2, 8, False) in group_keys       # mlp_model
    
    def test_group_models_with_shared_config(self, mock_raw_data, mock_feature_set_manager,
                                           sample_hyperparameters, models_with_same_data_config):
        """Test that models with same config are grouped together."""
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        groups = runner._group_models_by_data_config(models_with_same_data_config)
        
        # Should have 1 group (both models share same data config)
        assert len(groups) == 1
        
        # The group should contain both models
        shared_config = ("basic", 1, 10, False)
        assert shared_config in groups
        assert len(groups[shared_config]) == 2
    
    @patch('training.runner.DataProcessor')
    def test_process_data_for_config(self, mock_data_processor_class, mock_raw_data,
                                   mock_feature_set_manager, sample_hyperparameters, mock_data_processor):
        """Test data processing for specific configuration."""
        mock_data_processor_class.side_effect = mock_data_processor
        
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        config_key = ("basic", 1, 10, False)
        ml_data = runner._process_data_for_config(config_key)
        
        # Verify DataProcessor was called with correct parameters
        mock_data_processor_class.assert_called_once()
        call_args = mock_data_processor_class.call_args
        
        # Check that the right parameters were passed
        assert call_args[1]['forecast_period'] == 1
        assert call_args[1]['input_width'] == 10
        assert call_args[1]['normalise'] == False
        assert call_args[1]['train_ratio'] == 0.7
        assert call_args[1]['val_ratio'] == 0.15
    
    @patch('training.runner.DataProcessor')
    @patch('training.runner.ModelTrainer')
    def test_train_single_model(self, mock_trainer_class, mock_data_processor_class,
                               mock_raw_data, mock_feature_set_manager, sample_hyperparameters,
                               mock_data_processor):
        """Test training a single model with pre-processed data."""
        
        # Set up mocks
        mock_data_processor_class.side_effect = mock_data_processor
        
        mock_trainer = Mock()
        mock_trainer.train_model.return_value = Mock()  # Mock history
        mock_trainer.evaluate_model.return_value = {
            'metrics': {'mse': 1.5, 'mae': 1.0, 'rmse': 1.22, 'r2': 0.85},
            'predictions': [1.0, 2.0, 3.0, 4.0],
            'actuals': [1.1, 2.1, 2.9, 4.2]
        }
        mock_trainer_class.return_value = mock_trainer
        
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        # Create mock ml_data and processed_data
        ml_data = mock_data_processor(
            raw_data=mock_raw_data,
            feature_set=["feature1", "target"],
            forecast_period=1,
            input_width=10,
            normalise=False,
            train_ratio=0.7,
            val_ratio=0.15
        )
        processed_data = ml_data.get_data()
        
        model_config = {
            "name": "test_model",
            "model": "linear",
            "params": {"epochs": 2, "batch_size": 16}
        }
        
        result = runner._train_single_model(model_config, ml_data, processed_data)
        
        # Verify result structure
        assert 'dates' in result
        assert 'actual' in result
        assert 'predictions' in result
        assert 'stats' in result
        
        # Verify model was trained and evaluated
        mock_trainer.train_model.assert_called_once()
        mock_trainer.evaluate_model.assert_called_once()
        
        # Verify results content
        assert "test_model" in result['predictions']
        assert "test_model" in result['stats']
    
    @patch('training.runner.DataProcessor')
    @patch('training.runner.ModelTrainer')
    def test_run_models_single_model(self, mock_trainer_class, mock_data_processor_class,
                                   mock_raw_data, mock_feature_set_manager, sample_hyperparameters,
                                   single_model_list, mock_data_processor):
        """Test running a single model through the complete pipeline."""
        
        # Set up mocks
        mock_data_processor_class.side_effect = mock_data_processor
        
        mock_trainer = Mock()
        mock_trainer.train_model.return_value = Mock()
        mock_trainer.evaluate_model.return_value = {
            'metrics': {'mse': 1.0, 'mae': 0.8, 'rmse': 1.0, 'r2': 0.9},
            'predictions': [1.0, 2.0, 3.0],
            'actuals': [1.1, 2.1, 2.9]
        }
        mock_trainer_class.return_value = mock_trainer
        
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        results = runner.run_models(single_model_list)
        
        # Verify result structure
        assert 'dates' in results
        assert 'actual' in results
        assert 'predictions' in results
        assert 'stats' in results
        
        # Verify single model results
        assert len(results['predictions']) == 1
        assert len(results['stats']) == 1
        assert "test_linear" in results['predictions']
        assert "test_linear" in results['stats']
    
    @patch('training.runner.DataProcessor')
    @patch('training.runner.ModelTrainer')
    def test_run_models_multiple_models(self, mock_trainer_class, mock_data_processor_class,
                                      mock_raw_data, mock_feature_set_manager, sample_hyperparameters,
                                      multiple_model_list, mock_data_processor):
        """Test running multiple models with different configurations."""
        
        # Set up mocks
        mock_data_processor_class.side_effect = mock_data_processor
        
        def create_trainer(*args, **kwargs):
            trainer = Mock()
            trainer.train_model.return_value = Mock()
            trainer.evaluate_model.return_value = {
                'metrics': {
                    'mse': np.random.random(), 
                    'mae': np.random.random(), 
                    'rmse': np.random.random(), 
                    'r2': np.random.random()
                },
                'predictions': np.random.random(10).tolist(),
                'actuals': np.random.random(10).tolist()
            }
            return trainer
        
        mock_trainer_class.side_effect = create_trainer
        
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        results = runner.run_models(multiple_model_list)
        
        # Verify all models were processed
        assert len(results['predictions']) == 3
        assert len(results['stats']) == 3
        
        # Verify each model is represented
        expected_names = ["linear_model", "lstm_model", "mlp_model"]
        for name in expected_names:
            assert name in results['predictions']
            assert name in results['stats']
    
    @patch('training.runner.DataProcessor')
    def test_run_models_with_shared_data_config(self, mock_data_processor_class, mock_raw_data,
                                              mock_feature_set_manager, sample_hyperparameters,
                                              models_with_same_data_config, mock_data_processor):
        """Test that models with same data config share data processing."""
        
        mock_data_processor_class.side_effect = mock_data_processor
        
        with patch('training.runner.ModelTrainer') as mock_trainer_class:
            mock_trainer = Mock()
            mock_trainer.train_model.return_value = Mock()
            mock_trainer.evaluate_model.return_value = {
                'metrics': {'mse': 1.0, 'mae': 0.8, 'rmse': 1.0, 'r2': 0.9},
                'predictions': [1.0, 2.0],
                'actuals': [1.1, 2.1]
            }
            mock_trainer_class.return_value = mock_trainer
            
            runner = ModelRunner(
                raw_data=mock_raw_data,
                feature_set_manager=mock_feature_set_manager,
                hyperparameters=sample_hyperparameters
            )
            
            results = runner.run_models(models_with_same_data_config)
            
            # DataProcessor should only be called once (shared config)
            assert mock_data_processor_class.call_count == 1
            
            # But both models should be trained
            assert mock_trainer_class.call_count == 2
            
            # Both models should have results
            assert len(results['predictions']) == 2
            assert "linear_shared" in results['predictions']
            assert "mlp_shared" in results['predictions']
    
    @patch('training.runner.DataProcessor')
    @patch('training.runner.ModelTrainer')
    def test_error_handling_in_model_training(self, mock_trainer_class, mock_data_processor_class,
                                            mock_raw_data, mock_feature_set_manager, 
                                            sample_hyperparameters, single_model_list, mock_data_processor):
        """Test error handling when model training fails."""
        
        mock_data_processor_class.side_effect = mock_data_processor
        
        # Set up trainer to raise error
        mock_trainer = Mock()
        mock_trainer.train_model.side_effect = RuntimeError("Training failed!")
        mock_trainer_class.return_value = mock_trainer
        
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        results = runner.run_models(single_model_list)
        
        # Should handle error gracefully
        model_name = single_model_list[0]["name"]
        assert model_name in results['predictions']
        assert model_name in results['stats']
        assert results['predictions'][model_name] == []
        assert 'error' in results['stats'][model_name]
        assert 'Training failed!' in results['stats'][model_name]['error']
    
    def test_get_test_dates(self, mock_raw_data, mock_feature_set_manager, sample_hyperparameters):
        """Test test dates generation."""
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        # Create mock ml_data (not used in this method, but consistent with signature)
        mock_ml_data = Mock()
        dates = runner._get_test_dates(mock_ml_data)
        
        assert isinstance(dates, list)
        assert len(dates) > 0
        
        # Verify date format
        for date_str in dates[:5]:  # Check first 5 dates
            assert isinstance(date_str, str)
            # Should be in format "YYYY-MM-DD HH:MM:SS"
            assert len(date_str.split()) == 2  # Date and time parts
            assert len(date_str.split()[0].split('-')) == 3  # Year-Month-Day
    
    @patch('training.runner.DataProcessor')
    @patch('training.runner.ModelTrainer')
    def test_normalization_reversal(self, mock_trainer_class, mock_data_processor_class,
                                  mock_raw_data, mock_feature_set_manager, sample_hyperparameters,
                                  mock_data_processor):
        """Test that normalization is properly reversed."""
        
        # Set up normalized data processor
        def create_normalized_processor(*args, **kwargs):
            processor = mock_data_processor(*args, **kwargs)
            processor.get_normalise.return_value = True
            processor.get_normalisation_params.return_value = {
                "mean": 100.0, "std": 20.0
            }
            return processor
        
        mock_data_processor_class.side_effect = create_normalized_processor
        
        # Set up trainer with mock predictions
        mock_trainer = Mock()
        mock_trainer.train_model.return_value = Mock()
        normalized_predictions = np.array([0.5, 1.0, -0.5])  # Normalized values
        normalized_actuals = np.array([0.6, 0.9, -0.4])
        
        mock_trainer.evaluate_model.return_value = {
            'metrics': {'mse': 1.0, 'mae': 0.8, 'rmse': 1.0, 'r2': 0.9},
            'predictions': normalized_predictions.tolist(),
            'actuals': normalized_actuals.tolist()
        }
        mock_trainer_class.return_value = mock_trainer
        
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        model_list = [{
            "name": "test_model",
            "model": "linear",
            "featureSet": "basic",
            "forecastPeriod": 1,
            "inputWidth": 10,
            "normalise": True,  # Enable normalization
            "params": {"epochs": 2}
        }]
        
        results = runner.run_models(model_list)
        
        # Verify normalization was reversed
        expected_predictions = (normalized_predictions * 20.0) + 100.0  # Reverse normalization
        expected_actuals = (normalized_actuals * 20.0) + 100.0
        
        actual_predictions = np.array(results['predictions']['test_model'])
        actual_actuals = np.array(results['actual'])
        
        np.testing.assert_allclose(actual_predictions, expected_predictions, rtol=1e-10)
        np.testing.assert_allclose(actual_actuals, expected_actuals, rtol=1e-10)
    
    def test_feature_set_manager_integration(self, mock_raw_data, sample_hyperparameters):
        """Test integration with feature set manager."""
        feature_set_manager = Mock()
        
        # Test different feature sets
        feature_sets = {
            "basic": ["feature1", "target"],
            "advanced": ["feature1", "feature2", "target"],
            "minimal": ["target"]
        }
        
        def get_feature_set(name):
            if name not in feature_sets:
                raise ValueError(f"Unknown feature set: {name}")
            return feature_sets[name]
        
        feature_set_manager.get_feature_set.side_effect = get_feature_set
        
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        # Verify feature set manager is accessible
        assert runner.feature_set_manager is feature_set_manager
        
        # Test that feature set manager method works correctly
        basic_features = runner.feature_set_manager.get_feature_set("basic")
        assert basic_features == ["feature1", "target"]

    def test_runner_has_cache_attributes(self, mock_raw_data, mock_feature_set_manager, sample_hyperparameters):
        """Test that ModelRunner has caching attributes."""
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        # Should have cache attribute
        assert hasattr(runner, '_data_cache')
        assert isinstance(runner._data_cache, dict)
        assert len(runner._data_cache) == 0  # Initially empty

    def test_cache_methods_exist(self, mock_raw_data, mock_feature_set_manager, sample_hyperparameters):
        """Test that caching methods exist and are callable."""
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        # Should have all cache-related methods
        assert hasattr(runner, 'get_cache_info')
        assert callable(runner.get_cache_info)
        assert hasattr(runner, 'clear_cache')
        assert callable(runner.clear_cache)
        assert hasattr(runner, 'get_cache_memory_usage')
        assert callable(runner.get_cache_memory_usage)
        assert hasattr(runner, '_get_processed_data_cached')
        assert callable(runner._get_processed_data_cached)

class TestModelRunnerCaching:
    """Additional tests specifically for caching functionality."""
    
    @patch('training.runner.DataProcessor')
    def test_data_caching_mechanism(self, mock_data_processor_class, mock_raw_data, 
                                  mock_feature_set_manager, sample_hyperparameters, mock_data_processor):
        """Test that data processing is cached correctly."""
        
        mock_data_processor_class.side_effect = mock_data_processor
        
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        config_key = ("basic", 1, 10, False)
        
        # First call should process data
        data1 = runner._get_processed_data_cached(config_key)
        assert len(runner._data_cache) == 1
        
        # Second call with same parameters should use cache
        data2 = runner._get_processed_data_cached(config_key)
        assert data1 is data2  # Same object from cache
        assert len(runner._data_cache) == 1  # Still only one cached item
        
        # DataProcessor should only be called once
        assert mock_data_processor_class.call_count == 1
        
        # Different parameters should create new cache entry
        different_config = ("advanced", 1, 10, False)
        data3 = runner._get_processed_data_cached(different_config)
        assert data3 is not data1  # Different object
        assert len(runner._data_cache) == 2  # Now two cached items
        assert mock_data_processor_class.call_count == 2
    
    @patch('training.runner.DataProcessor')
    @patch('training.runner.ModelTrainer')
    def test_run_single_model_with_caching(self, mock_trainer_class, mock_data_processor_class,
                                         mock_raw_data, mock_feature_set_manager, 
                                         sample_hyperparameters, mock_data_processor):
        """Test run_single_model method with caching."""
        
        # Set up mocks
        mock_data_processor_class.side_effect = mock_data_processor
        
        mock_trainer = Mock()
        mock_trainer.train_model.return_value = Mock()
        mock_trainer.evaluate_model.return_value = {
            'metrics': {'mse': 1.5, 'mae': 1.0, 'rmse': 1.22, 'r2': 0.85},
            'predictions': [1.0, 2.0, 3.0, 4.0],
            'actuals': [1.1, 2.1, 2.9, 4.2]
        }
        mock_trainer_class.return_value = mock_trainer
        
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        model_config = {
            "name": "test_linear",
            "model": "linear",
            "featureSet": "basic",
            "forecastPeriod": 1,
            "inputWidth": 10,
            "normalise": False,
            "params": {"epochs": 3, "batch_size": 16}
        }
        
        # First run
        result1 = runner.run_single_model(model_config)
        assert mock_data_processor_class.call_count == 1
        assert len(runner._data_cache) == 1
        
        # Second run with same config should use cache
        result2 = runner.run_single_model(model_config)
        assert mock_data_processor_class.call_count == 1  # Still only called once
        assert len(runner._data_cache) == 1  # Cache size unchanged
        
        # Verify both results have correct structure
        for result in [result1, result2]:
            assert 'dates' in result
            assert 'actual' in result
            assert 'predictions' in result
            assert 'stats' in result
            assert "test_linear" in result['predictions']
    
    # @patch('training.runner.DataProcessor')
    # @patch('training.runner.ModelTrainer')
    # def test_cache_across_batch_and_single_processing(self, mock_trainer_class, mock_data_processor_class,
    #                                                 mock_raw_data, mock_feature_set_manager, 
    #                                                 sample_hyperparameters, mock_data_processor):
    #     """Test that cache works across both batch and single model processing."""
        
    #     # Set up mocks
    #     mock_data_processor_class.side_effect = mock_data_processor
        
    #     mock_trainer = Mock()
    #     mock_trainer.train_model.return_value = Mock()
    #     mock_trainer.evaluate_model.return_value = {
    #         'metrics': {'mse': 1.0, 'mae': 0.8, 'rmse': 1.0, 'r2': 0.9},
    #         'predictions': [1.0, 2.0, 3.0],
    #         'actuals': [1.1, 2.1, 2.9]
    #     }
    #     mock_trainer_class.return_value = mock_trainer
        
    #     runner = ModelRunner(
    #         raw_data=mock_raw_data,
    #         feature_set_manager=mock_feature_set_manager,
    #         hyperparameters=sample_hyperparameters
    #     )
        
    #     # Batch processing first
    #     batch_models = [{
    #         "name": "batch_model",
    #         "model": "linear",
    #         "featureSet": "basic",
    #         "forecastPeriod": 1,
    #         "inputWidth": 10,
    #         "normalise": False,
    #         "params": {"epochs": 2}
    #     }]        
    #     batch_results = runner.run_models(batch_models)
        
    #     assert len(runner._data_cache) > 0
    #     initial_cache_size = len(runner._data_cache)
    #     initial_processor_calls = mock_data_processor_class.call_count
        
    #     # Single model processing with same config should use cache
    #     single_model = {
    #         "name": "single_model",
    #         "model": "mlp",  # Different model, same data config
    #         "featureSet": "basic",      # Same
    #         "forecastPeriod": 1,        # Same
    #         "inputWidth": 10,           # Same
    #         "normalise": False,         # Same
    #         "params": {"epochs": 2, "hidden_units": [32]}
    #     }
        
    #     single_result = runner.run_single_model(single_model)

    #     # Cache should be reused
    #     assert len(runner._data_cache) == initial_cache_size
    #     assert mock_data_processor_class.call_count == initial_processor_calls
        
    #     # Results should be valid
    #     assert "single_model" in single_result['predictions']
    #     assert "batch_model" in batch_results['predictions']
    
    def test_cache_info_method(self, mock_raw_data, mock_feature_set_manager, sample_hyperparameters):
        """Test cache info method for debugging."""
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        # Initially empty
        info = runner.get_cache_info()
        assert info['cached_configurations'] == 0
        assert info['cache_keys'] == []
        
        # Add some mock cache entries
        runner._data_cache[('basic', 1, 10, False)] = Mock()
        runner._data_cache[('advanced', 2, 15, True)] = Mock()
        
        info = runner.get_cache_info()
        assert info['cached_configurations'] == 2
        assert len(info['cache_keys']) == 2
        assert ('basic', 1, 10, False) in info['cache_keys']
        assert ('advanced', 2, 15, True) in info['cache_keys']
    
    def test_clear_cache_method(self, mock_raw_data, mock_feature_set_manager, sample_hyperparameters):
        """Test cache clearing functionality."""
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters,
            verbose=False
        )
        
        # Add some cache entries
        runner._data_cache[('basic', 1, 10, False)] = Mock()
        runner._data_cache[('advanced', 2, 15, True)] = Mock()
        
        assert len(runner._data_cache) == 2
        
        # Clear cache
        runner.clear_cache()
        
        assert len(runner._data_cache) == 0
        
        # Cache info should reflect empty state
        info = runner.get_cache_info()
        assert info['cached_configurations'] == 0
    
    def test_cache_memory_usage_estimation(self, mock_raw_data, mock_feature_set_manager, sample_hyperparameters):
        """Test cache memory usage estimation."""
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        # Create mock ml_data with realistic data
        mock_ml_data = Mock()
        mock_processed_data = {
            'x_train': np.random.randn(100, 10, 3),  # Some realistic size
            'y_train': np.random.randn(100, 1),
            'x_val': np.random.randn(50, 10, 3),
            'y_val': np.random.randn(50, 1),
            'x_test': np.random.randn(50, 10, 3),
            'y_test': np.random.randn(50, 1)
        }
        mock_ml_data.get_data.return_value = mock_processed_data
        
        # Add to cache
        runner._data_cache[('basic', 1, 10, False)] = mock_ml_data
        
        # Get memory usage
        usage = runner.get_cache_memory_usage()
        
        assert 'estimated_bytes' in usage
        assert 'estimated_mb' in usage
        assert 'cached_configs' in usage
        assert usage['cached_configs'] == 1
        assert usage['estimated_bytes'] > 0
        assert usage['estimated_mb'] > 0
    
    @patch('training.runner.DataProcessor')
    def test_cache_with_different_configurations(self, mock_data_processor_class, mock_raw_data,
                                               mock_feature_set_manager, sample_hyperparameters, mock_data_processor):
        """Test caching behavior with various configuration combinations."""
        
        mock_data_processor_class.side_effect = mock_data_processor
        
        runner = ModelRunner(
            raw_data=mock_raw_data,
            feature_set_manager=mock_feature_set_manager,
            hyperparameters=sample_hyperparameters
        )
        
        # Test different configuration combinations
        configs = [
            ("basic", 1, 10, False),
            ("basic", 1, 10, True),      # Different normalization
            ("basic", 1, 15, False),     # Different input width
            ("basic", 2, 10, False),     # Different forecast period
            ("advanced", 1, 10, False),  # Different feature set
        ]
        
        # Process each config
        for config in configs:
            runner._get_processed_data_cached(config)
        
        # Should have separate cache entries for each
        assert len(runner._data_cache) == len(configs)
        assert mock_data_processor_class.call_count == len(configs)
        
        # Processing same config again should use cache
        runner._get_processed_data_cached(configs[0])
        assert mock_data_processor_class.call_count == len(configs)  # No additional calls


