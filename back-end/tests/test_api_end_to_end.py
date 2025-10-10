import pytest
import sys
import os
import json
import numpy as np
import pandas as pd
import tempfile

# Add parent directory to path (following your pattern)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app

# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def client():
    """Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def realistic_api_data():
    """Realistic time series data in API format."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    
    # Generate realistic price data
    base_price = 100.0
    price_changes = np.random.normal(0, 2, 100)
    prices = base_price + np.cumsum(price_changes)
    
    # Create correlated features
    volumes = 1000 + np.random.randint(-100, 100, 100)
    
    return [
        {
            'date': date.strftime('%Y-%m-%d'),
            'price': float(price),
            'volume': int(volume),
            'target': float(price)
        }
        for date, price, volume in zip(dates, prices, volumes)
    ]

@pytest.fixture
def minimal_api_data():
    """Minimal valid API data for quick tests."""
    return [
        {'date': '2023-01-01', 'price': 100.0, 'target': 100.0},
        {'date': '2023-01-02', 'price': 101.0, 'target': 101.0},
        {'date': '2023-01-03', 'price': 102.0, 'target': 102.0},
        {'date': '2023-01-04', 'price': 103.0, 'target': 103.0},
        {'date': '2023-01-05', 'price': 104.0, 'target': 104.0},
        {'date': '2023-01-06', 'price': 105.0, 'target': 105.0},
        {'date': '2023-01-07', 'price': 106.0, 'target': 106.0},
        {'date': '2023-01-08', 'price': 107.0, 'target': 107.0},
        {'date': '2023-01-09', 'price': 108.0, 'target': 108.0},
        {'date': '2023-01-10', 'price': 109.0, 'target': 109.0}
    ]

@pytest.fixture
def single_model_request():
    """Single model API request."""
    return {
        "name": "api_test_linear",
        "model": "linear",
        "featureSet": "basic",
        "forecastPeriod": 1,
        "inputWidth": 5,
        "normalise": False,
        "params": {
            "epochs": 2,
            "batch_size": 8
        }
    }

@pytest.fixture
def multiple_models_request():
    """Multiple models API request."""
    return [
        {
            "name": "api_linear",
            "model": "linear",
            "featureSet": "basic",
            "forecastPeriod": 1,
            "inputWidth": 5,
            "normalise": False,
            "params": {"epochs": 2}
        },
        {
            "name": "api_baseline",
            "model": "baseline",
            "featureSet": "basic",
            "forecastPeriod": 1,
            "inputWidth": 5,
            "normalise": False,
            "params": {"epochs": 1}
        }
    ]

@pytest.fixture
def api_hyperparameters():
    """API hyperparameters."""
    return {
        "train_val_test_split": [0.6, 0.2, 0.2]
    }

# =============================================================================
# API END-TO-END TESTS
# =============================================================================

class TestAPIEndToEnd:
    """End-to-end API integration tests."""
    
    def test_api_single_model_success(self, client, realistic_api_data, 
                                     single_model_request, api_hyperparameters):
        """Test successful single model API call end-to-end."""
        
        request_data = {
            "rawData": realistic_api_data,
            "modelList": [single_model_request],
            "hyperparameters": api_hyperparameters
        }
        
        response = client.post('/api/run_models',
                              json=request_data,
                              headers={'Content-Type': 'application/json'})
        
        # Verify HTTP response
        assert response.status_code == 200
        assert response.content_type == 'application/json'
        
        # Parse response data
        data = response.get_json()
        
        # Verify response structure
        assert isinstance(data, dict)
        assert data['success'] == True
        assert 'dates' in data
        assert 'actual' in data
        assert 'predictions' in data
        assert 'stats' in data
        assert 'metadata' in data
        
        # Verify model results
        model_name = single_model_request['name']
        assert model_name in data['predictions']
        assert model_name in data['stats']
        
        # Verify data integrity
        predictions = data['predictions'][model_name]
        actual = data['actual']
        dates = data['dates']
        
        assert len(predictions) == len(actual) == len(dates)
        assert all(isinstance(p, (int, float)) for p in predictions)
        assert all(isinstance(a, (int, float)) for a in actual)
        assert all(isinstance(d, str) for d in dates)
        
        # Verify stats
        stats = data['stats'][model_name]
        required_metrics = ['mse', 'mae', 'rmse', 'r2']
        for metric in required_metrics:
            assert metric in stats
            assert isinstance(stats[metric], (int, float))
        
        # Verify metadata
        metadata = data['metadata']
        assert metadata['total_models'] == 1
        assert metadata['successful_models'] == 1
        assert metadata['failed_models'] == 0
    
    def test_api_multiple_models_success(self, client, realistic_api_data,
                                        multiple_models_request, api_hyperparameters):
        """Test successful multiple models API call end-to-end."""
        
        request_data = {
            "rawData": realistic_api_data,
            "modelList": multiple_models_request,
            "hyperparameters": api_hyperparameters
        }
        
        response = client.post('/api/run_models',
                              json=request_data,
                              headers={'Content-Type': 'application/json'})
        
        assert response.status_code == 200
        data = response.get_json()
        
        # Verify multiple models processed
        assert data['success'] == True
        assert len(data['predictions']) == 2
        assert len(data['stats']) == 2
        
        # Verify each model
        for model_config in multiple_models_request:
            model_name = model_config['name']
            assert model_name in data['predictions']
            assert model_name in data['stats']
            assert len(data['predictions'][model_name]) > 0
        
        # Verify metadata
        assert data['metadata']['total_models'] == 2
        assert data['metadata']['successful_models'] == 2
    
    def test_api_validation_errors(self, client):
        """Test API validation error responses."""
        
        # Test no JSON data
        response = client.post('/api/run_models')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        
        # Test empty model list
        request_data = {
            "rawData": [{'date': '2023-01-01', 'price': 100, 'target': 100}],
            "modelList": [],
            "hyperparameters": {}
        }
        
        response = client.post('/api/run_models', json=request_data)
        assert response.status_code == 400
        data = response.get_json()
        assert 'No models provided' in data['error']
        
        # Test empty data
        request_data = {
            "rawData": [],
            "modelList": [{"name": "test", "model": "linear"}],
            "hyperparameters": {}
        }
        
        response = client.post('/api/run_models', json=request_data)
        assert response.status_code == 400
        data = response.get_json()
        assert 'No data provided' in data['error']
    
    def test_api_partial_model_failure(self, client, realistic_api_data, api_hyperparameters):
        """Test API handling of partial model failures."""
        
        # Mix valid and invalid models
        mixed_models = [
            {
                "name": "api_valid_model",
                "model": "linear",
                "featureSet": "basic",
                "forecastPeriod": 1,
                "inputWidth": 5,
                "normalise": False,
                "params": {"epochs": 2}
            },
            {
                "name": "api_invalid_model",
                "model": "nonexistent_model",
                "featureSet": "basic",
                "forecastPeriod": 1,
                "inputWidth": 5,
                "normalise": False,
                "params": {"epochs": 2}
            }
        ]
        
        request_data = {
            "rawData": realistic_api_data,
            "modelList": mixed_models,
            "hyperparameters": api_hyperparameters
        }
        
        response = client.post('/api/run_models', json=request_data)
        
        # Should return 200 with partial success
        assert response.status_code == 200
        data = response.get_json()
        
        # Should indicate partial success
        assert 'warning' in data
        assert 'Only 1/2 models succeeded' in data['warning']
        
        # Valid model should succeed
        assert 'api_valid_model' in data['predictions']
        assert len(data['predictions']['api_valid_model']) > 0
        assert 'error' not in data['stats']['api_valid_model']
        
        # Invalid model should fail gracefully
        assert 'api_invalid_model' in data['predictions']
        assert data['predictions']['api_invalid_model'] == []
        assert 'error' in data['stats']['api_invalid_model']
    
    def test_api_all_models_fail(self, client, realistic_api_data, api_hyperparameters):
        """Test API response when all models fail."""
        
        failing_models = [{
            "name": "failing_model",
            "model": "nonexistent_model",
            "featureSet": "basic",
            "forecastPeriod": 1,
            "inputWidth": 5,
            "normalise": False,
            "params": {"epochs": 2}
        }]
        
        request_data = {
            "rawData": realistic_api_data,
            "modelList": failing_models,
            "hyperparameters": api_hyperparameters
        }
        
        response = client.post('/api/run_models', json=request_data)
        
        # Should return 500 when all models fail
        assert response.status_code == 500
        data = response.get_json()
        assert 'All models failed to process' in data['error']
    
    def test_api_cors_preflight(self, client):
        """Test CORS preflight handling."""
        
        response = client.options('/api/run_models')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'
    
    def test_api_invalid_json(self, client):
        """Test API handling of invalid JSON."""
        
        response = client.post('/api/run_models',
                              data='invalid json',
                              headers={'Content-Type': 'application/json'})
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'Invalid JSON' in data['error']
    
    def test_api_verbose_mode(self, client, minimal_api_data, single_model_request,
                             api_hyperparameters):
        """Test API verbose mode via query parameter."""
        
        request_data = {
            "rawData": minimal_api_data,
            "modelList": [single_model_request],
            "hyperparameters": api_hyperparameters
        }
        
        response = client.post('/api/run_models?verbose=1',
                              json=request_data,
                              headers={'Content-Type': 'application/json'})
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] == True
        
        # Verbose mode doesn't change response structure, just logging
        assert 'predictions' in data
        assert 'stats' in data
    
    def test_api_different_hyperparameters(self, client, realistic_api_data, single_model_request):
        """Test API with different hyperparameter configurations."""
        
        custom_hyperparameters = {
            "train_val_test_split": [0.5, 0.3, 0.2],
            "custom_param": "test_value"
        }
        
        request_data = {
            "rawData": realistic_api_data,
            "modelList": [single_model_request],
            "hyperparameters": custom_hyperparameters
        }
        
        response = client.post('/api/run_models', json=request_data)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] == True
        
        # Should process successfully with custom hyperparameters
        model_name = single_model_request['name']
        assert model_name in data['predictions']
        assert len(data['predictions'][model_name]) > 0
    
    def test_api_normalization_end_to_end(self, client, realistic_api_data, api_hyperparameters):
        """Test API normalization workflow end-to-end."""
        
        # Model with normalization
        normalized_model = {
            "name": "api_normalized",
            "model": "linear",
            "featureSet": "basic",
            "forecastPeriod": 1,
            "inputWidth": 5,
            "normalise": True,
            "params": {"epochs": 2}
        }
        
        request_data = {
            "rawData": realistic_api_data,
            "modelList": [normalized_model],
            "hyperparameters": api_hyperparameters
        }
        
        response = client.post('/api/run_models', json=request_data)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] == True
        
        # Verify normalization worked (predictions should be reasonable)
        predictions = data['predictions']['api_normalized']
        assert len(predictions) > 0
        assert all(isinstance(p, (int, float)) for p in predictions)
        assert all(np.isfinite(p) for p in predictions)
    
    def test_api_response_consistency(self, client, realistic_api_data, multiple_models_request,
                                    api_hyperparameters):
        """Test API response format consistency."""
        
        request_data = {
            "rawData": realistic_api_data,
            "modelList": multiple_models_request,
            "hyperparameters": api_hyperparameters
        }
        
        response = client.post('/api/run_models', json=request_data)
        
        assert response.status_code == 200
        data = response.get_json()
        
        # Verify consistent response structure
        required_keys = ['success', 'dates', 'actual', 'predictions', 'stats', 'metadata']
        for key in required_keys:
            assert key in data
        
        # Verify data types
        assert isinstance(data['success'], bool)
        assert isinstance(data['dates'], list)
        assert isinstance(data['actual'], list)
        assert isinstance(data['predictions'], dict)
        assert isinstance(data['stats'], dict)
        assert isinstance(data['metadata'], dict)
        
        # Verify metadata structure
        metadata_keys = ['total_models', 'successful_models', 'failed_models', 'cache_info']
        for key in metadata_keys:
            assert key in data['metadata']
    
    def test_api_large_dataset_handling(self, client, api_hyperparameters):
        """Test API handling of larger datasets."""
        
        # Create larger dataset
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=500, freq="D")
        prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
        
        large_dataset = [
            {
                'date': date.strftime('%Y-%m-%d'),
                'price': float(price),
                'target': float(price)
            }
            for date, price in zip(dates, prices)
        ]
        
        model_request = {
            "name": "large_dataset_test",
            "model": "linear",
            "featureSet": "basic",
            "forecastPeriod": 1,
            "inputWidth": 10,
            "normalise": True,
            "params": {"epochs": 3}
        }
        
        request_data = {
            "rawData": large_dataset,
            "modelList": [model_request],
            "hyperparameters": api_hyperparameters
        }
        
        response = client.post('/api/run_models', json=request_data)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] == True
        
        # Should handle larger datasets successfully
        predictions = data['predictions']['large_dataset_test']
        assert len(predictions) > 0
        
        # Verify reasonable processing (larger test set)
        assert len(data['actual']) > 20  # Should have reasonable test set size
    
    def test_api_concurrent_request_simulation(self, client, realistic_api_data,
                                             single_model_request, api_hyperparameters):
        """Simulate concurrent API requests (sequential for testing)."""
        
        request_data = {
            "rawData": realistic_api_data,
            "modelList": [single_model_request],
            "hyperparameters": api_hyperparameters
        }
        
        # Simulate multiple concurrent requests
        responses = []
        for i in range(3):
            response = client.post('/api/run_models', json=request_data)
            responses.append(response)
        
        # All requests should succeed
        for i, response in enumerate(responses):
            assert response.status_code == 200
            data = response.get_json()
            assert data['success'] == True
            
            model_name = single_model_request['name']
            assert model_name in data['predictions']
            assert len(data['predictions'][model_name]) > 0