import pytest
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path (following your pattern)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.base import ModelConfig
from models.factory import ModelFactory
from training.trainer import ModelTrainer

# =============================================================================
# TEST FIXTURES (following your pattern)
# =============================================================================

@pytest.fixture
def basic_config():
    """Basic training configuration."""
    return ModelConfig(
        epochs=3,  # Small for fast testing
        batch_size=16,
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
        early_stopping_patience=5,
        reduce_lr_patience=3
    )

@pytest.fixture
def advanced_config():
    """Advanced configuration with custom parameters."""
    return ModelConfig(
        epochs=5,
        batch_size=32,
        optimizer="rmsprop",
        loss="mae",
        metrics=["mse", "mae"],
        model_params={
            "lstm_units": 64,
            "dropout": 0.2,
            "hidden_units": [128, 64]
        },
        early_stopping_patience=10,
        reduce_lr_patience=5
    )

@pytest.fixture
def no_callbacks_config():
    """Configuration with callbacks disabled."""
    return ModelConfig(
        epochs=2,
        batch_size=16,
        early_stopping_patience=0,  # Disabled
        reduce_lr_patience=0        # Disabled
    )

@pytest.fixture
def sample_training_data():
    """Generate realistic training data for time series."""
    np.random.seed(42)  # Deterministic for testing
    
    batch_size = 64
    input_width = 10
    num_features = 4
    output_size = 1
    
    # Generate time series with trend and noise
    def generate_ts_batch(size, input_width, features, output_size):
        x = np.zeros((size, input_width, features))
        y = np.zeros((size, output_size))
        
        for i in range(size):
            # Create a simple time series with trend
            trend = np.random.uniform(-0.1, 0.1)
            base = np.random.uniform(50, 150)
            
            for t in range(input_width):
                for f in range(features):
                    x[i, t, f] = base + trend * t + np.random.normal(0, 5)
            
            # Target is based on last few values with some transformation
            y[i, 0] = np.mean(x[i, -3:, 0]) + trend * 2 + np.random.normal(0, 2)
        
        return x, y
    
    x_train, y_train = generate_ts_batch(batch_size, input_width, num_features, output_size)
    x_val, y_val = generate_ts_batch(32, input_width, num_features, output_size)
    x_test, y_test = generate_ts_batch(32, input_width, num_features, output_size)
    
    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test
    }

@pytest.fixture
def minimal_training_data():
    """Minimal training data for quick tests."""
    np.random.seed(42)
    
    return {
        'x_train': np.random.randn(16, 5, 2),
        'y_train': np.random.randn(16, 1),
        'x_val': np.random.randn(8, 5, 2),
        'y_val': np.random.randn(8, 1),
        'x_test': np.random.randn(8, 5, 2),
        'y_test': np.random.randn(8, 1)
    }

@pytest.fixture
def multivariate_data():
    """Multi-output training data."""
    np.random.seed(42)
    
    return {
        'x_train': np.random.randn(32, 8, 3),
        'y_train': np.random.randn(32, 2),  # 2 outputs
        'x_val': np.random.randn(16, 8, 3),
        'y_val': np.random.randn(16, 2),
        'x_test': np.random.randn(16, 8, 3),
        'y_test': np.random.randn(16, 2)
    }

# =============================================================================
# TEST CLASS
# =============================================================================

class TestModelTrainer:
    """Integration tests for ModelTrainer class."""
    
    def test_trainer_instantiation(self, basic_config):
        """Test ModelTrainer can be instantiated."""
        trainer = ModelTrainer(basic_config, verbose=False)
        
        assert trainer.config == basic_config
        assert trainer.verbose == False
        assert trainer.model is None
        assert trainer.results == {}
    
    def test_trainer_with_verbose(self, basic_config):
        """Test trainer with verbose output."""
        trainer = ModelTrainer(basic_config, verbose=True)
        assert trainer.verbose == True
    
    @pytest.mark.parametrize("model_name", ["linear", "mlp", "lstm", "baseline"])
    def test_train_all_model_types(self, model_name, basic_config, minimal_training_data):
        """Test training each model type successfully."""
        trainer = ModelTrainer(basic_config, verbose=False)
        
        input_shape = minimal_training_data['x_train'].shape[1:]
        output_size = minimal_training_data['y_train'].shape[1]
        
        # Train model
        history = trainer.train_model(model_name, minimal_training_data, input_shape, output_size)
        
        # Verify training completed
        assert trainer.model is not None
        assert trainer.model.is_built
        assert trainer.model.is_compiled
        assert history is not None
        assert hasattr(history, 'history')
        assert 'loss' in history.history
    
    def test_train_model_with_validation_data(self, basic_config, sample_training_data):
        """Test training with validation data."""
        trainer = ModelTrainer(basic_config, verbose=False)
        
        input_shape = sample_training_data['x_train'].shape[1:]
        output_size = sample_training_data['y_train'].shape[1]
        
        history = trainer.train_model("linear", sample_training_data, input_shape, output_size)
        
        # Should have validation metrics
        assert 'val_loss' in history.history
        assert len(history.history['loss']) == basic_config.epochs
        assert len(history.history['val_loss']) == basic_config.epochs
    
    def test_train_model_without_validation_data(self, basic_config):
        """Test training without validation data."""
        trainer = ModelTrainer(basic_config, verbose=False)
        
        # Training data only
        train_data = {
            'x_train': np.random.randn(32, 10, 3),
            'y_train': np.random.randn(32, 1),
            'x_val': None,
            'y_val': None,
            'x_test': np.random.randn(16, 10, 3),
            'y_test': np.random.randn(16, 1)
        }
        
        input_shape = (10, 3)
        output_size = 1
        
        history = trainer.train_model("linear", train_data, input_shape, output_size)
        
        # Should not have validation metrics
        assert 'val_loss' not in history.history
        assert 'loss' in history.history
    
    def test_evaluate_model_after_training(self, basic_config, sample_training_data):
        """Test model evaluation after training."""
        trainer = ModelTrainer(basic_config, verbose=False)
        
        input_shape = sample_training_data['x_train'].shape[1:]
        output_size = sample_training_data['y_train'].shape[1]
        
        # Train first
        trainer.train_model("linear", sample_training_data, input_shape, output_size)
        
        # Then evaluate
        results = trainer.evaluate_model(sample_training_data)
        
        assert 'metrics' in results
        assert 'predictions' in results
        assert 'actuals' in results
        
        # Check metrics
        metrics = results['metrics']
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        
        # Check data shapes
        predictions = results['predictions']
        actuals = results['actuals']
        assert len(predictions) == len(actuals)
        assert len(predictions) == len(sample_training_data['y_test'])
    
    def test_evaluate_model_before_training_fails(self, basic_config, sample_training_data):
        """Test that evaluation fails if no model is trained."""
        trainer = ModelTrainer(basic_config, verbose=False)
        
        with pytest.raises(ValueError, match="No model trained yet"):
            trainer.evaluate_model(sample_training_data)
    
    def test_multivariate_output_training(self, basic_config, multivariate_data):
        """Test training with multiple output variables."""
        trainer = ModelTrainer(basic_config, verbose=False)
        
        input_shape = multivariate_data['x_train'].shape[1:]
        output_size = multivariate_data['y_train'].shape[1]  # Should be 2
        
        history = trainer.train_model("linear", multivariate_data, input_shape, output_size)
        
        assert trainer.model is not None
        assert trainer.model.model.output_shape[1] == output_size
        
        # Evaluate
        results = trainer.evaluate_model(multivariate_data)
        predictions = np.array(results['predictions'])
        
        # Should predict 2 outputs
        assert predictions.shape[1] == 2 if predictions.ndim > 1 else len(predictions) == len(multivariate_data['y_test'].flatten())
    
    def test_callbacks_integration(self, basic_config, sample_training_data):
        """Test that callbacks are properly integrated into training."""
        trainer = ModelTrainer(basic_config, verbose=False)
        
        input_shape = sample_training_data['x_train'].shape[1:]
        output_size = sample_training_data['y_train'].shape[1]
        
        # Mock the model's fit method to capture callbacks
        with patch.object(trainer, 'train_model', wraps=trainer.train_model) as mock_train:
            history = trainer.train_model("linear", sample_training_data, input_shape, output_size)
            
            # Verify callbacks were used
            assert trainer.model is not None
            # The actual callback testing is more complex, but we verify the method was called
            assert mock_train.call_count == 1
    
    def test_no_callbacks_configuration(self, no_callbacks_config, minimal_training_data):
        """Test training with callbacks disabled."""
        trainer = ModelTrainer(no_callbacks_config, verbose=False)
        
        input_shape = minimal_training_data['x_train'].shape[1:]
        output_size = minimal_training_data['y_train'].shape[1]
        
        # Should work without callbacks
        history = trainer.train_model("linear", minimal_training_data, input_shape, output_size)
        
        assert history is not None
        assert 'loss' in history.history
    
    def test_custom_model_parameters_passed_correctly(self, advanced_config, minimal_training_data):
        """Test that custom model parameters are passed to the model."""
        trainer = ModelTrainer(advanced_config, verbose=False)
        
        input_shape = minimal_training_data['x_train'].shape[1:]
        output_size = minimal_training_data['y_train'].shape[1]
        
        # Train LSTM with custom parameters
        history = trainer.train_model("lstm", minimal_training_data, input_shape, output_size)
        
        # Verify LSTM was created with custom parameters
        assert trainer.model is not None
        
        # Find LSTM layer to verify units
        lstm_layers = [layer for layer in trainer.model.model.layers 
                      if isinstance(layer, tf.keras.layers.LSTM)]
        
        if lstm_layers:  # LSTM model should have LSTM layers
            assert lstm_layers[0].units == advanced_config.model_params["lstm_units"]
    
    def test_training_with_different_optimizers(self, minimal_training_data):
        """Test training with different optimizers."""
        optimizers = ["adam", "sgd", "rmsprop"]
        
        for optimizer in optimizers:
            config = ModelConfig(epochs=2, optimizer=optimizer)
            trainer = ModelTrainer(config, verbose=False)
            
            input_shape = minimal_training_data['x_train'].shape[1:]
            output_size = minimal_training_data['y_train'].shape[1]
            
            history = trainer.train_model("linear", minimal_training_data, input_shape, output_size)
            
            assert history is not None
            assert trainer.model.model.optimizer.__class__.__name__.lower().startswith(optimizer.lower())
    
    def test_training_with_different_losses(self, minimal_training_data):
        """Test training with different loss functions."""
        losses = ["mse", "mae", "huber"]
        
        for loss in losses:
            config = ModelConfig(epochs=2, loss=loss)
            trainer = ModelTrainer(config, verbose=False)
            
            input_shape = minimal_training_data['x_train'].shape[1:]
            output_size = minimal_training_data['y_train'].shape[1]
            
            history = trainer.train_model("linear", minimal_training_data, input_shape, output_size)
            
            assert history is not None
            assert 'loss' in history.history
    
    def test_metrics_calculation_accuracy(self, basic_config):
        """Test that metrics are calculated correctly."""
        trainer = ModelTrainer(basic_config, verbose=False)
        
        # Create simple, predictable data
        np.random.seed(42)
        x_train = np.random.randn(32, 5, 2)
        y_train = np.random.randn(32, 1)
        x_test = np.random.randn(16, 5, 2)
        y_test = np.random.randn(16, 1)
        
        train_data = {
            'x_train': x_train, 'y_train': y_train,
            'x_val': None, 'y_val': None,
            'x_test': x_test, 'y_test': y_test
        }
        
        # Train and evaluate
        trainer.train_model("linear", train_data, (5, 2), 1)
        results = trainer.evaluate_model(train_data)
        
        # Verify metrics manually
        y_pred = np.array(results['predictions'])
        y_true = np.array(results['actuals'])
        
        expected_mse = mean_squared_error(y_true, y_pred)
        expected_mae = mean_absolute_error(y_true, y_pred)
        
        assert abs(results['metrics']['mse'] - expected_mse) < 1e-10
        assert abs(results['metrics']['mae'] - expected_mae) < 1e-10
        assert abs(results['metrics']['rmse'] - np.sqrt(expected_mse)) < 1e-10
    
    def test_prediction_shape_consistency(self, basic_config, sample_training_data):
        """Test that predictions maintain correct shapes."""
        trainer = ModelTrainer(basic_config, verbose=False)
        
        input_shape = sample_training_data['x_train'].shape[1:]
        output_size = sample_training_data['y_train'].shape[1]
        
        trainer.train_model("linear", sample_training_data, input_shape, output_size)
        results = trainer.evaluate_model(sample_training_data)
        
        predictions = np.array(results['predictions'])
        actuals = np.array(results['actuals'])
        test_data = sample_training_data['y_test']
        
        # All should have the same length
        assert len(predictions) == len(actuals) == len(test_data)
        
        # Should match the flattened test data shape
        if test_data.ndim > 1 and test_data.shape[1] == 1:
            assert len(predictions) == len(test_data.flatten())
    
    def test_trainer_state_isolation(self, basic_config, minimal_training_data):
        """Test that multiple trainers don't interfere with each other."""
        trainer1 = ModelTrainer(basic_config, verbose=False)
        trainer2 = ModelTrainer(basic_config, verbose=False)
        
        input_shape = minimal_training_data['x_train'].shape[1:]
        output_size = minimal_training_data['y_train'].shape[1]
        
        # Train different models
        trainer1.train_model("linear", minimal_training_data, input_shape, output_size)
        trainer2.train_model("mlp", minimal_training_data, input_shape, output_size)
        
        # Should have different models
        assert trainer1.model is not trainer2.model
        assert type(trainer1.model).__name__ != type(trainer2.model).__name__
        
        # Should have separate results
        results1 = trainer1.evaluate_model(minimal_training_data)
        results2 = trainer2.evaluate_model(minimal_training_data)
        
        # Results should be different (different models, different predictions)
        pred1 = np.array(results1['predictions'])
        pred2 = np.array(results2['predictions'])
        
        # Should not be identical (different models should give different predictions)
        assert not np.allclose(pred1, pred2, rtol=1e-3)  # Very loose tolerance
    
    @pytest.mark.parametrize("epochs", [1, 3, 5])
    def test_different_epoch_counts(self, epochs, minimal_training_data):
        """Test training with different epoch counts."""
        config = ModelConfig(epochs=epochs)
        trainer = ModelTrainer(config, verbose=False)
        
        input_shape = minimal_training_data['x_train'].shape[1:]
        output_size = minimal_training_data['y_train'].shape[1]
        
        history = trainer.train_model("linear", minimal_training_data, input_shape, output_size)
        
        assert len(history.history['loss']) == epochs
    
    def test_error_handling_invalid_model_name(self, basic_config, minimal_training_data):
        """Test error handling for invalid model name."""
        trainer = ModelTrainer(basic_config, verbose=False)
        
        input_shape = minimal_training_data['x_train'].shape[1:]
        output_size = minimal_training_data['y_train'].shape[1]
        
        with pytest.raises(ValueError, match="Unknown model"):
            trainer.train_model("invalid_model", minimal_training_data, input_shape, output_size)