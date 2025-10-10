import pytest
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

# Add parent directory to path (following your pattern)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.model_service import process_models_request

# =============================================================================
# TEST FIXTURES FOR SERVICE LAYER
# =============================================================================

@pytest.fixture
def sample_raw_data_list():
    """Sample raw data as list of dictionaries (API format)."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    
    # Simple time series data
    np.random.seed(42)
    values = 100 + np.cumsum(np.random.randn(100) * 0.5)  # Random walk
    
    return [
        {
            'date': date.strftime('%Y-%m-%d'),
            'price': value,
            'volume': 1000 + np.random.randint(-100, 100),
            'feature1': value + np.random.randn() * 2,
            'target': value
        }
        for date, value in zip(dates, values)
    ]

@pytest.fixture
def minimal_raw_data_list():
    """Minimal valid raw data for quick tests."""
    return [
        {'date': '2023-01-01', 'price': 100, 'target': 100},
        {'date': '2023-01-02', 'price': 101, 'target': 101},
        {'date': '2023-01-03', 'price': 102, 'target': 102},
        {'date': '2023-01-04', 'price': 103, 'target': 103},
        {'date': '2023-01-05', 'price': 104, 'target': 104},
        {'date': '2023-01-06', 'price': 105, 'target': 105},
        {'date': '2023-01-07', 'price': 106, 'target': 106},
        {'date': '2023-01-08', 'price': 107, 'target': 107},
        {'date': '2023-01-09', 'price': 108, 'target': 108},
        {'date': '2023-01-10', 'price': 109, 'target': 109}
    ]

@pytest.fixture
def sample_single_model_list():
    """Single model configuration for service testing."""
    return [{
        "name": "service_test_linear",
        "model": "linear",
        "featureSet": "basic",
        "forecastPeriod": 1,
        "inputWidth": 5,
        "normalise": False,
        "params": {
            "epochs": 2,
            "batch_size": 8
        }
    }]

@pytest.fixture
def sample_multiple_models_list():
    """Multiple model configurations for service testing."""
    return [
        {
            "name": "service_linear",
            "model": "linear",
            "featureSet": "basic",
            "forecastPeriod": 1,
            "inputWidth": 5,
            "normalise": False,
            "params": {"epochs": 2, "batch_size": 8}
        },
        {
            "name": "service_mlp",
            "model": "mlp",
            "featureSet": "basic",
            "forecastPeriod": 1,
            "inputWidth": 5,
            "normalise": True,
            "params": {"epochs": 2, "hidden_units": [16]}
        }
    ]

@pytest.fixture
def sample_hyperparameters():
    """Sample hyperparameters for service testing."""
    return {
        "train_val_test_split": [0.6, 0.2, 0.2],
        "other_param": "test_value"
    }

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
                {"function": "raw_data", "input_data_fields": ["feature1"], "name": "feature1"},
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
def invalid_raw_data_samples():
    """Various invalid raw data formats for error testing."""
    return {
        'empty_list': [],
        'missing_date': [{'price': 100, 'target': 100}],
        'invalid_structure': [{'invalid': 'data'}],
        'non_dict_items': ['invalid', 'data'],
        'mixed_types': [{'date': '2023-01-01', 'price': 'invalid'}]
    }

# =============================================================================
# SERVICE LAYER TESTS
# =============================================================================

class TestModelService:
    """Tests for the model service business logic."""
    
    @patch('services.model_service.ModelRunner')
    @patch('services.model_service.FeatureSetManager')
    def test_process_models_request_success_single_model(self, mock_fsm_class, mock_runner_class,
                                                        sample_raw_data_list, sample_single_model_list,
                                                        sample_hyperparameters, mock_feature_sets_file):
        """Test successful processing of single model request."""
        
        # Mock FeatureSetManager
        mock_fsm = Mock()
        mock_fsm_class.return_value = mock_fsm
        
        # Mock ModelRunner
        mock_runner = Mock()
        mock_runner.run_single_model.return_value = {
            'dates': ['2023-01-01', '2023-01-02'],
            'actual': [100.5, 101.2],
            'predictions': {'service_test_linear': [100.3, 101.1]},
            'stats': {'service_test_linear': {'mse': 0.05, 'mae': 0.04, 'rmse': 0.22, 'r2': 0.95}}
        }
        mock_runner.get_cache_info.return_value = {'cached_configurations': 1, 'cache_keys': []}
        mock_runner_class.return_value = mock_runner
        
        # Process request
        results = process_models_request(
            raw_data_list=sample_raw_data_list,
            model_list=sample_single_model_list,
            hyperparameters=sample_hyperparameters,
            feature_sets_file=mock_feature_sets_file,
            verbose=True
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert results['success'] == True
        assert 'dates' in results
        assert 'actual' in results
        assert 'predictions' in results
        assert 'stats' in results
        assert 'errors' in results
        assert 'metadata' in results
        
        # Verify content
        assert len(results['dates']) == 2
        assert len(results['actual']) == 2
        assert 'service_test_linear' in results['predictions']
        assert 'service_test_linear' in results['stats']
        
        # Verify metadata
        metadata = results['metadata']
        assert metadata['total_models'] == 1
        assert metadata['successful_models'] == 1
        assert metadata['failed_models'] == 0
        
        # Verify mocks were called correctly
        mock_fsm_class.assert_called_once_with(mock_feature_sets_file)
        mock_runner_class.assert_called_once()
        mock_runner.run_single_model.assert_called_once_with(sample_single_model_list[0])
    
    @patch('services.model_service.ModelRunner')
    @patch('services.model_service.FeatureSetManager')
    def test_process_models_request_success_multiple_models(self, mock_fsm_class, mock_runner_class,
                                                           sample_raw_data_list, sample_multiple_models_list,
                                                           sample_hyperparameters, mock_feature_sets_file):
        """Test successful processing of multiple models."""
        
        # Mock FeatureSetManager
        mock_fsm = Mock()
        mock_fsm_class.return_value = mock_fsm
        
        # Mock ModelRunner with different results for each model
        mock_runner = Mock()
        
        def mock_run_single_model(config):
            model_name = config['name']
            return {
                'dates': ['2023-01-01', '2023-01-02'],
                'actual': [100.0, 101.0],
                'predictions': {model_name: [99.8, 100.9]},
                'stats': {model_name: {'mse': 0.1, 'mae': 0.08, 'rmse': 0.32, 'r2': 0.90}}
            }
        
        mock_runner.run_single_model.side_effect = mock_run_single_model
        mock_runner.get_cache_info.return_value = {'cached_configurations': 1, 'cache_keys': []}
        mock_runner_class.return_value = mock_runner
        
        # Process request
        results = process_models_request(
            raw_data_list=sample_raw_data_list,
            model_list=sample_multiple_models_list,
            hyperparameters=sample_hyperparameters,
            feature_sets_file=mock_feature_sets_file,
            verbose=False
        )
        
        # Verify results
        assert results['success'] == True
        assert len(results['predictions']) == 2
        assert len(results['stats']) == 2
        
        # Verify both models are represented
        assert 'service_linear' in results['predictions']
        assert 'service_mlp' in results['predictions']
        assert 'service_linear' in results['stats']
        assert 'service_mlp' in results['stats']
        
        # Verify metadata
        assert results['metadata']['total_models'] == 2
        assert results['metadata']['successful_models'] == 2
        assert results['metadata']['failed_models'] == 0
        
        # Verify ModelRunner was called for each model
        assert mock_runner.run_single_model.call_count == 2
    
    def test_process_models_request_validation_no_models(self, sample_raw_data_list, 
                                                        sample_hyperparameters, mock_feature_sets_file):
        """Test validation error when no models provided."""
        
        with pytest.raises(ValueError, match="No models provided"):
            process_models_request(
                raw_data_list=sample_raw_data_list,
                model_list=[],  # Empty model list
                hyperparameters=sample_hyperparameters,
                feature_sets_file=mock_feature_sets_file
            )
    
    def test_process_models_request_validation_no_data(self, sample_single_model_list,
                                                      sample_hyperparameters, mock_feature_sets_file):
        """Test validation error when no data provided."""
        
        with pytest.raises(ValueError, match="No data provided"):
            process_models_request(
                raw_data_list=[],  # Empty data list
                model_list=sample_single_model_list,
                hyperparameters=sample_hyperparameters,
                feature_sets_file=mock_feature_sets_file
            )
    
    def test_process_models_request_validation_invalid_data_format(self, sample_single_model_list,
                                                                  sample_hyperparameters, mock_feature_sets_file,
                                                                  invalid_raw_data_samples):
        """Test validation errors for invalid data formats."""
        
        for data_type, invalid_data in invalid_raw_data_samples.items():
            with pytest.raises(ValueError, match="No models provided|No data provided|Data must contain 'date' column|Data is empty|Invalid data format|Invalid numeric values found in 'price' column"):
                process_models_request(
                    raw_data_list=invalid_data,
                    model_list=sample_single_model_list,
                    hyperparameters=sample_hyperparameters,
                    feature_sets_file=mock_feature_sets_file
                )
    
    def test_process_models_request_validation_missing_date_column(self, sample_single_model_list,
                                                                  sample_hyperparameters, mock_feature_sets_file):
        """Test validation error when date column is missing."""
        
        invalid_data = [
            {'price': 100, 'target': 100},  # Missing 'date'
            {'price': 101, 'target': 101}
        ]
        
        with pytest.raises(ValueError, match="Data must contain 'date' column"):
            process_models_request(
                raw_data_list=invalid_data,
                model_list=sample_single_model_list,
                hyperparameters=sample_hyperparameters,
                feature_sets_file=mock_feature_sets_file
            )
    
    @patch('services.model_service.FeatureSetManager')
    def test_process_models_request_feature_set_manager_error(self, mock_fsm_class, sample_raw_data_list,
                                                             sample_single_model_list, sample_hyperparameters):
        """Test error handling when FeatureSetManager fails."""
        
        # Mock FeatureSetManager to raise error
        mock_fsm_class.side_effect = FileNotFoundError("Feature sets file not found")
        
        with pytest.raises(ValueError, match="Failed to load feature sets"):
            process_models_request(
                raw_data_list=sample_raw_data_list,
                model_list=sample_single_model_list,
                hyperparameters=sample_hyperparameters,
                feature_sets_file="nonexistent_file.json"
            )
    
    @patch('services.model_service.ModelRunner')
    @patch('services.model_service.FeatureSetManager')
    def test_process_models_request_model_runner_initialization_error(self, mock_fsm_class, mock_runner_class,
                                                                     sample_raw_data_list, sample_single_model_list,
                                                                     sample_hyperparameters, mock_feature_sets_file):
        """Test error handling when ModelRunner initialization fails."""
        
        # Mock FeatureSetManager
        mock_fsm_class.return_value = Mock()
        
        # Mock ModelRunner to raise error during initialization
        mock_runner_class.side_effect = RuntimeError("ModelRunner initialization failed")
        
        with pytest.raises(ValueError, match="Failed to initialize ModelRunner"):
            process_models_request(
                raw_data_list=sample_raw_data_list,
                model_list=sample_single_model_list,
                hyperparameters=sample_hyperparameters,
                feature_sets_file=mock_feature_sets_file
            )
    
    @patch('services.model_service.ModelRunner')
    @patch('services.model_service.FeatureSetManager')
    def test_process_models_request_partial_model_failures(self, mock_fsm_class, mock_runner_class,
                                                          sample_raw_data_list, sample_hyperparameters, 
                                                          mock_feature_sets_file):
        """Test handling when some models succeed and some fail."""
        
        # Mock FeatureSetManager
        mock_fsm_class.return_value = Mock()
        
        # Mixed model list with valid and invalid models
        mixed_models = [
            {
                "name": "working_model",
                "model": "linear",
                "featureSet": "basic",
                "forecastPeriod": 1,
                "inputWidth": 5,
                "normalise": False,
                "params": {"epochs": 2}
            },
            {
                "name": "failing_model",
                "model": "invalid_model_type",
                "featureSet": "basic",
                "forecastPeriod": 1,
                "inputWidth": 5,
                "normalise": False,
                "params": {"epochs": 2}
            }
        ]
        
        # Mock ModelRunner
        mock_runner = Mock()
        
        def mock_run_single_model(config):
            if config['name'] == 'working_model':
                return {
                    'dates': ['2023-01-01'],
                    'actual': [100.0],
                    'predictions': {'working_model': [99.5]},
                    'stats': {'working_model': {'mse': 0.25, 'mae': 0.5, 'rmse': 0.5, 'r2': 0.8}}
                }
            else:
                raise ValueError("Invalid model type")
        
        mock_runner.run_single_model.side_effect = mock_run_single_model
        mock_runner.get_cache_info.return_value = {'cached_configurations': 1, 'cache_keys': []}
        mock_runner_class.return_value = mock_runner
        
        # Process request - should not raise exception
        results = process_models_request(
            raw_data_list=sample_raw_data_list,
            model_list=mixed_models,
            hyperparameters=sample_hyperparameters,
            feature_sets_file=mock_feature_sets_file
        )
        
        # Verify partial success
        assert results['success'] == True  # At least one model succeeded
        assert len(results['predictions']) == 2
        assert len(results['stats']) == 2
        assert len(results['errors']) == 1
        
        # Verify working model succeeded
        assert 'working_model' in results['predictions']
        assert results['predictions']['working_model'] == [99.5]
        assert 'error' not in results['stats']['working_model']
        
        # Verify failing model was handled
        assert 'failing_model' in results['predictions']
        assert results['predictions']['failing_model'] == []
        assert 'error' in results['stats']['failing_model']
        assert 'failing_model' in results['errors']
        
        # Verify metadata
        assert results['metadata']['total_models'] == 2
        assert results['metadata']['successful_models'] == 1
        assert results['metadata']['failed_models'] == 1
    
    @patch('services.model_service.ModelRunner')
    @patch('services.model_service.FeatureSetManager')
    def test_process_models_request_all_models_fail(self, mock_fsm_class, mock_runner_class,
                                                   sample_raw_data_list, sample_hyperparameters,
                                                   mock_feature_sets_file):
        """Test error handling when all models fail."""
        
        # Mock FeatureSetManager
        mock_fsm_class.return_value = Mock()
        
        # Mock ModelRunner to always fail
        mock_runner = Mock()
        mock_runner.run_single_model.side_effect = RuntimeError("All models failing")
        mock_runner.get_cache_info.return_value = {'cached_configurations': 0, 'cache_keys': []}
        mock_runner_class.return_value = mock_runner
        
        failing_models = [{
            "name": "failing_model",
            "model": "linear",
            "featureSet": "basic",
            "forecastPeriod": 1,
            "inputWidth": 5,
            "normalise": False,
            "params": {"epochs": 2}
        }]
        
        with pytest.raises(Exception, match="All models failed to process"):
            process_models_request(
                raw_data_list=sample_raw_data_list,
                model_list=failing_models,
                hyperparameters=sample_hyperparameters,
                feature_sets_file=mock_feature_sets_file
            )
    
    @patch('services.model_service.ModelRunner')
    @patch('services.model_service.FeatureSetManager')
    def test_process_models_request_verbose_mode(self, mock_fsm_class, mock_runner_class,
                                                minimal_raw_data_list, sample_single_model_list,
                                                sample_hyperparameters, mock_feature_sets_file,
                                                capsys):
        """Test verbose mode prints processing messages."""
        
        # Mock FeatureSetManager
        mock_fsm_class.return_value = Mock()
        
        # Mock ModelRunner
        mock_runner = Mock()
        mock_runner.run_single_model.return_value = {
            'dates': ['2023-01-01'],
            'actual': [100.0],
            'predictions': {'service_test_linear': [99.5]},
            'stats': {'service_test_linear': {'mse': 0.25, 'mae': 0.5, 'rmse': 0.5, 'r2': 0.8}}
        }
        mock_runner.get_cache_info.return_value = {'cached_configurations': 1, 'cache_keys': []}
        mock_runner_class.return_value = mock_runner
        
        # Process with verbose=True
        process_models_request(
            raw_data_list=minimal_raw_data_list,
            model_list=sample_single_model_list,
            hyperparameters=sample_hyperparameters,
            feature_sets_file=mock_feature_sets_file,
            verbose=True
        )
        
        # Verify verbose output
        captured = capsys.readouterr()
        assert "Processing model: service_test_linear" in captured.out
    
    @patch('services.model_service.ModelRunner')
    @patch('services.model_service.FeatureSetManager')
    def test_process_models_request_hyperparameters_passed(self, mock_fsm_class, mock_runner_class,
                                                          sample_raw_data_list, sample_single_model_list,
                                                          mock_feature_sets_file):
        """Test that hyperparameters are correctly passed to ModelRunner."""
        
        # Mock FeatureSetManager
        mock_fsm_class.return_value = Mock()
        
        # Mock ModelRunner
        mock_runner = Mock()
        mock_runner.run_single_model.return_value = {
            'dates': ['2023-01-01'],
            'actual': [100.0],
            'predictions': {'service_test_linear': [99.5]},
            'stats': {'service_test_linear': {'mse': 0.25, 'mae': 0.5, 'rmse': 0.5, 'r2': 0.8}}
        }
        mock_runner.get_cache_info.return_value = {'cached_configurations': 1, 'cache_keys': []}
        mock_runner_class.return_value = mock_runner
        
        custom_hyperparameters = {
            "train_val_test_split": [0.5, 0.3, 0.2],
            "custom_param": "custom_value"
        }
        
        # Process request
        process_models_request(
            raw_data_list=sample_raw_data_list,
            model_list=sample_single_model_list,
            hyperparameters=custom_hyperparameters,
            feature_sets_file=mock_feature_sets_file
        )
        
        # Verify ModelRunner was called with correct hyperparameters
        call_args = mock_runner_class.call_args
        assert call_args[1]['hyperparameters'] == custom_hyperparameters
    
    @patch('services.model_service.ModelRunner')  
    @patch('services.model_service.FeatureSetManager')
    def test_process_models_request_dataframe_conversion(self, mock_fsm_class, mock_runner_class,
                                                        sample_single_model_list, sample_hyperparameters,
                                                        mock_feature_sets_file):
        """Test that raw data list is correctly converted to DataFrame."""
        
        # Mock FeatureSetManager
        mock_fsm_class.return_value = Mock()
        
        # Mock ModelRunner
        mock_runner = Mock()
        mock_runner.run_single_model.return_value = {
            'dates': ['2023-01-01'],
            'actual': [100.0],
            'predictions': {'service_test_linear': [99.5]},
            'stats': {'service_test_linear': {'mse': 0.25, 'mae': 0.5, 'rmse': 0.5, 'r2': 0.8}}
        }
        mock_runner.get_cache_info.return_value = {'cached_configurations': 1, 'cache_keys': []}
        mock_runner_class.return_value = mock_runner
        
        # Specific raw data structure
        test_data = [
            {'date': '2023-01-01', 'price': 100, 'volume': 1000, 'target': 100},
            {'date': '2023-01-02', 'price': 101, 'volume': 1100, 'target': 101}
        ]
        
        # Process request
        process_models_request(
            raw_data_list=test_data,
            model_list=sample_single_model_list,
            hyperparameters=sample_hyperparameters,
            feature_sets_file=mock_feature_sets_file
        )
        
        # Verify ModelRunner was called with DataFrame
        call_args = mock_runner_class.call_args
        raw_data_df = call_args[1]['raw_data']
        
        assert isinstance(raw_data_df, pd.DataFrame)
        assert len(raw_data_df) == 2
        assert list(raw_data_df.columns) == ['date', 'price', 'volume', 'target']
        assert raw_data_df.iloc[0]['price'] == 100
        assert raw_data_df.iloc[1]['price'] == 101