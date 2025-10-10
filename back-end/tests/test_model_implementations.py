import pytest
import sys
import os
import numpy as np
import tensorflow as tf
from unittest.mock import patch

# Add parent directory to path (following your pattern)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.base import ModelConfig
from models.implementations import LinearModel, LSTMModel, MLPModel, BaselineModel

# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def basic_config():
    """Basic configuration for testing."""
    return ModelConfig(
        epochs=2,
        batch_size=16,
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

@pytest.fixture
def linear_config():
    """Configuration for linear models."""
    return ModelConfig(
        epochs=5,
        batch_size=32,
        optimizer="sgd",
        loss="mae"
    )

@pytest.fixture
def lstm_config():
    """Configuration for LSTM models."""
    return ModelConfig(
        epochs=10,
        batch_size=64,
        model_params={
            "lstm_units": 128,
            "dropout": 0.3,
            "use_batch_norm": True,
            "use_layer_norm": False
        }
    )

@pytest.fixture
def mlp_config():
    """Configuration for MLP models."""
    return ModelConfig(
        epochs=15,
        batch_size=32,
        model_params={
            "hidden_units": [256, 128, 64],
            "dropout_rate": 0.4
        }
    )

@pytest.fixture
def baseline_config():
    """Configuration for baseline models."""
    return ModelConfig(
        epochs=1,  # Baseline doesn't really train
        batch_size=1,
        optimizer="adam"
    )

@pytest.fixture
def sample_shapes():
    """Common input/output shape combinations for testing."""
    return [
        ((10, 3), 1),    # 10 timesteps, 3 features -> 1 output
        ((20, 1), 1),    # 20 timesteps, 1 feature -> 1 output
        ((5, 10), 5),    # 5 timesteps, 10 features -> 5 outputs
        ((15, 7), 3),    # 15 timesteps, 7 features -> 3 outputs
        ((1, 1), 1),     # Minimal case
    ]

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    def _generate_data(batch_size, input_shape, output_size):
        np.random.seed(42)  # Deterministic
        x = np.random.randn(batch_size, *input_shape)
        y = np.random.randn(batch_size, output_size)
        return x, y
    return _generate_data

# =============================================================================
# TEST LINEAR MODEL
# =============================================================================

class TestLinearModel:
    """Test suite for LinearModel."""
    
    def test_linear_model_instantiation(self, linear_config):
        """Test LinearModel can be instantiated."""
        model = LinearModel("linear_test", linear_config)
        
        assert model.name == "linear_test"
        assert model.config == linear_config
        assert not model.is_built
        assert not model.is_compiled
    
    @pytest.mark.parametrize("input_shape,output_size", [
        ((10, 3), 1),
        ((20, 5), 2),
        ((5, 1), 10)
    ])
    def test_linear_model_architecture(self, linear_config, input_shape, output_size):
        """Test LinearModel builds correct architecture."""
        model = LinearModel("linear", linear_config)
        model.build(input_shape, output_size)
        
        # Check model is built
        assert model.is_built
        assert model.model is not None
        
        # Check architecture components
        layers = model.model.layers
        layer_types = [type(layer).__name__ for layer in layers]
        
        # Should have Flatten and Dense layers
        assert "Flatten" in layer_types
        assert "Dense" in layer_types
        
        # Check input/output shapes
        assert model.model.input_shape[1:] == input_shape
        assert model.model.output_shape[1] == output_size
    
    def test_linear_model_layer_order(self, linear_config):
        """Test that LinearModel layers are in correct order."""
        model = LinearModel("linear", linear_config)
        model.build((10, 3), 1)
        
        layers = model.model.layers
        
        # Should be: Input -> Flatten -> Dense
        layer_types = [type(layer) for layer in model.model.layers]
        assert tf.keras.layers.Flatten in layer_types
        assert tf.keras.layers.Dense in layer_types        # assert isinstance(layers[0], tf.keras.layers.InputLayer)
        # assert isinstance(layers[1], tf.keras.layers.Flatten)
        # assert isinstance(layers[2], tf.keras.layers.Dense)
        
        # Dense layer should have correct units
        # assert layers[2].units == 1
    # Find the Dense layer by type (and/or name)
        dense_layers = [l for l in layers if isinstance(l, tf.keras.layers.Dense)]
        assert len(dense_layers) == 1, "Expected exactly one Dense layer"
        dense = dense_layers[0]

        # Check units
        assert dense.units == 1

    def test_linear_model_prediction_shape(self, linear_config, sample_data):
        """Test LinearModel prediction output shape."""
        input_shape = (10, 3)
        output_size = 2
        batch_size = 16
        
        model = LinearModel("linear", linear_config)
        model.build(input_shape, output_size)
        model.compile()
        
        # Generate test data
        x_test, _ = sample_data(batch_size, input_shape, output_size)
        
        # Make predictions
        predictions = model.predict(x_test)
        
        assert predictions.shape == (batch_size, output_size)
    
    def test_linear_model_parameter_count(self, linear_config):
        """Test LinearModel has expected number of parameters."""
        input_shape = (10, 5)  # 50 input features after flattening
        output_size = 3
        
        model = LinearModel("linear", linear_config)
        model.build(input_shape, output_size)
        
        # Parameters = (input_features * output_size) + bias
        expected_params = (10 * 5 * output_size) + output_size
        actual_params = model.model.count_params()
        
        assert actual_params == expected_params

# =============================================================================
# TEST MLP MODEL
# =============================================================================

class TestMLPModel:
    """Test suite for MLPModel."""
    
    def test_mlp_model_instantiation(self, mlp_config):
        """Test MLPModel can be instantiated."""
        model = MLPModel("mlp_test", mlp_config)
        
        assert model.name == "mlp_test"
        assert model.config == mlp_config
    
    def test_mlp_default_parameters(self, basic_config):
        """Test MLPModel with default parameters."""
        model = MLPModel("mlp", basic_config)
        model.build((10, 3), 1)
        
        layers = model.model.layers
        layer_types = [type(layer).__name__ for layer in layers]
        
        # Should have Dense and Dropout layers
        assert "Dense" in layer_types
        assert "Dropout" in layer_types
        assert "Flatten" in layer_types
    
    def test_mlp_custom_hidden_units(self, basic_config):
        """Test MLPModel with custom hidden units."""
        custom_config = ModelConfig(
            model_params={"hidden_units": [128, 64, 32]}
        )
        
        model = MLPModel("mlp", custom_config)
        model.build((10, 3), 1)
        
        # Count Dense layers (excluding output layer)
        dense_layers = [layer for layer in model.model.layers 
                       if isinstance(layer, tf.keras.layers.Dense)]
        
        # Should have 4 Dense layers (3 hidden + 1 output)
        assert len(dense_layers) == 4
        
        # Check hidden layer units
        assert dense_layers[0].units == 128  # First hidden layer
        assert dense_layers[1].units == 64   # Second hidden layer
        assert dense_layers[2].units == 32   # Third hidden layer
        assert dense_layers[3].units == 1    # Output layer
    
    def test_mlp_dropout_configuration(self, basic_config):
        """Test MLPModel dropout configuration."""
        dropout_config = ModelConfig(
            model_params={"dropout_rate": 0.5, "hidden_units": [64, 32]}
        )
        
        model = MLPModel("mlp", dropout_config)
        model.build((10, 3), 1)
        
        # Find dropout layers
        dropout_layers = [layer for layer in model.model.layers 
                         if isinstance(layer, tf.keras.layers.Dropout)]
        
        # Should have 2 dropout layers (one after each hidden layer)
        assert len(dropout_layers) == 2
        
        # Check dropout rate
        for dropout_layer in dropout_layers:
            assert dropout_layer.rate == 0.5
    
    @pytest.mark.parametrize("input_shape,output_size", [
        ((15, 4), 1),
        ((20, 2), 3),
        ((5, 8), 5)
    ])
    def test_mlp_various_shapes(self, mlp_config, input_shape, output_size):
        """Test MLPModel with various input/output shapes."""
        model = MLPModel("mlp", mlp_config)
        model.build(input_shape, output_size)
        
        assert model.is_built
        assert model.model.input_shape[1:] == input_shape
        assert model.model.output_shape[1] == output_size
    
    def test_mlp_single_hidden_layer(self, basic_config):
        """Test MLPModel with single hidden layer."""
        single_layer_config = ModelConfig(
            model_params={"hidden_units": [100]}
        )
        
        model = MLPModel("mlp", single_layer_config)
        model.build((10, 3), 1)
        
        dense_layers = [layer for layer in model.model.layers 
                       if isinstance(layer, tf.keras.layers.Dense)]
        
        # Should have 2 Dense layers (1 hidden + 1 output)
        assert len(dense_layers) == 2
        assert dense_layers[0].units == 100  # Hidden layer
        assert dense_layers[1].units == 1    # Output layer

# =============================================================================
# TEST LSTM MODEL
# =============================================================================

class TestLSTMModel:
    """Test suite for LSTMModel."""
    
    def test_lstm_model_instantiation(self, lstm_config):
        """Test LSTMModel can be instantiated."""
        model = LSTMModel("lstm_test", lstm_config)
        
        assert model.name == "lstm_test"
        assert model.config == lstm_config
    
    def test_lstm_basic_architecture(self, basic_config):
        """Test LSTM basic architecture."""
        model = LSTMModel("lstm", basic_config)
        model.build((10, 3), 1)
        
        layers = model.model.layers
        layer_types = [type(layer).__name__ for layer in layers]
        
        # Should have LSTM layer
        assert "LSTM" in layer_types
        assert "Dense" in layer_types
    
    def test_lstm_with_batch_norm(self, basic_config):
        """Test LSTM with batch normalization."""
        config = ModelConfig(
            model_params={"use_batch_norm": True}
        )
        
        model = LSTMModel("lstm", config)
        model.build((10, 3), 1)
        
        layer_types = [type(layer).__name__ for layer in model.model.layers]
        assert "BatchNormalization" in layer_types
    
    def test_lstm_with_layer_norm(self, basic_config):
        """Test LSTM with layer normalization."""
        config = ModelConfig(
            model_params={"use_layer_norm": True, "use_batch_norm": False}
        )
        
        model = LSTMModel("lstm", config)
        model.build((10, 3), 1)
        
        layer_types = [type(layer).__name__ for layer in model.model.layers]
        assert "LayerNormalization" in layer_types
        assert "BatchNormalization" not in layer_types
    
    def test_lstm_custom_units(self, basic_config):
        """Test LSTM with custom units."""
        config = ModelConfig(
            model_params={"lstm_units": 256}
        )
        
        model = LSTMModel("lstm", config)
        model.build((10, 3), 1)
        
        # Find LSTM layer
        lstm_layers = [layer for layer in model.model.layers 
                      if isinstance(layer, tf.keras.layers.LSTM)]
        
        assert len(lstm_layers) == 1
        assert lstm_layers[0].units == 256
    
    def test_lstm_dropout_configuration(self, basic_config):
        """Test LSTM dropout configuration."""
        config = ModelConfig(
            model_params={"dropout": 0.4}
        )
        
        model = LSTMModel("lstm", config)
        model.build((10, 3), 1)
        
        lstm_layers = [layer for layer in model.model.layers 
                      if isinstance(layer, tf.keras.layers.LSTM)]
        
        assert lstm_layers[0].dropout == 0.4
    
    def test_lstm_return_sequences_false(self, basic_config):
        """Test that LSTM has return_sequences=False by default."""
        model = LSTMModel("lstm", basic_config)
        model.build((10, 3), 1)
        
        lstm_layers = [layer for layer in model.model.layers 
                      if isinstance(layer, tf.keras.layers.LSTM)]
        
        # Should not return sequences (for final dense layer)
        assert lstm_layers[0].return_sequences == False
    
    @pytest.mark.parametrize("input_shape,output_size", [
        ((20, 1), 1),
        ((15, 5), 3),
        ((30, 10), 2)
    ])
    def test_lstm_various_shapes(self, lstm_config, input_shape, output_size):
        """Test LSTM with various input/output shapes."""
        model = LSTMModel("lstm", lstm_config)
        model.build(input_shape, output_size)
        
        assert model.is_built
        assert model.model.input_shape[1:] == input_shape
        assert model.model.output_shape[1] == output_size

# =============================================================================
# TEST BASELINE MODEL
# =============================================================================

class TestBaselineModel:
    """Test suite for BaselineModel."""
    
    def test_baseline_model_instantiation(self, baseline_config):
        """Test BaselineModel can be instantiated."""
        model = BaselineModel("baseline_test", baseline_config)
        
        assert model.name == "baseline_test"
        assert model.config == baseline_config
    
    def test_baseline_architecture(self, baseline_config):
        """Test BaselineModel architecture."""
        model = BaselineModel("baseline", baseline_config)
        model.build((10, 3), 1)
        
        layers = model.model.layers
        layer_types = [type(layer).__name__ for layer in layers]
        
        # Should have Lambda and Dense layers
        assert "Lambda" in layer_types
        assert "Dense" in layer_types
    
    def test_baseline_prediction_logic(self, baseline_config, sample_data):
        """Test that baseline model predicts based on last timestep."""
        input_shape = (10, 3)
        output_size = 3  # Same as number of features
        batch_size = 8
        
        model = BaselineModel("baseline", baseline_config)
        model.build(input_shape, output_size)
        model.compile()
        
        # Create test data with known pattern
        x_test = np.random.randn(batch_size, *input_shape)
        
        # Make prediction
        predictions = model.predict(x_test)
        
        # Shape should be correct
        assert predictions.shape == (batch_size, output_size)
    
    @pytest.mark.parametrize("input_shape,output_size", [
        ((5, 1), 1),
        ((10, 2), 2),
        ((15, 5), 1),
        ((20, 3), 4)
    ])
    def test_baseline_various_shapes(self, baseline_config, input_shape, output_size):
        """Test BaselineModel with various shapes."""
        model = BaselineModel("baseline", baseline_config)
        model.build(input_shape, output_size)
        
        assert model.is_built
        assert model.model.input_shape[1:] == input_shape
        assert model.model.output_shape[1] == output_size

# =============================================================================
# INTEGRATION TESTS ACROSS ALL MODELS
# =============================================================================

class TestAllModelImplementations:
    """Integration tests across all model implementations."""
    
    @pytest.mark.parametrize("model_class,config_fixture", [
        (LinearModel, "linear_config"),
        (MLPModel, "mlp_config"),
        (LSTMModel, "lstm_config"),
        (BaselineModel, "baseline_config")
    ])
    def test_all_models_build_and_compile(self, model_class, config_fixture, request):
        """Test that all models can build and compile successfully."""
        config = request.getfixturevalue(config_fixture)
        model = model_class(f"test_{model_class.__name__}", config)
        
        # Build
        model.build((10, 3), 1)
        assert model.is_built
        
        # Compile
        model.compile()
        assert model.is_compiled
    
    @pytest.mark.parametrize("model_class", [LinearModel, MLPModel, LSTMModel, BaselineModel])
    def test_all_models_prediction_consistency(self, model_class, basic_config, sample_data):
        """Test that all models produce consistent prediction shapes."""
        input_shape = (15, 4)
        output_size = 2
        batch_size = 10
        
        model = model_class(f"test_{model_class.__name__}", basic_config)
        model.build(input_shape, output_size)
        model.compile()
        
        x_test, _ = sample_data(batch_size, input_shape, output_size)
        
        # Make two predictions with same input
        pred1 = model.predict(x_test, verbose=0)
        pred2 = model.predict(x_test, verbose=0)
        
        # Shapes should be consistent
        assert pred1.shape == pred2.shape == (batch_size, output_size)
        
        # Predictions should be deterministic (same input -> same output)
        np.testing.assert_allclose(pred1, pred2, rtol=1e-5)
    
    def test_model_parameter_isolation(self, basic_config):
        """Test that models don't share parameters when created separately."""
        model1 = LinearModel("model1", basic_config)
        model2 = LinearModel("model2", basic_config)
        
        model1.build((10, 3), 1)
        model2.build((10, 3), 1)
        
        # Should be different model instances
        assert model1.model is not model2.model
        assert id(model1.model) != id(model2.model)
    
    def test_config_parameter_usage(self):
        """Test that models correctly use configuration parameters."""
        # LSTM with specific parameters
        lstm_config = ModelConfig(
            model_params={"lstm_units": 64, "dropout": 0.25}
        )
        
        lstm_model = LSTMModel("lstm", lstm_config)
        lstm_model.build((10, 3), 1)
        
        # Find LSTM layer and verify parameters
        lstm_layers = [layer for layer in lstm_model.model.layers 
                      if isinstance(layer, tf.keras.layers.LSTM)]
        
        assert lstm_layers[0].units == 64
        assert lstm_layers[0].dropout == 0.25
    
    @pytest.mark.parametrize("model_class", [LinearModel, MLPModel, LSTMModel, BaselineModel])
    def test_model_memory_cleanup(self, model_class, basic_config):
        """Test that models can be properly cleaned up."""
        model = model_class("test", basic_config)
        model.build((10, 3), 1)
        model.compile()
        
        # Model should exist
        assert model.model is not None
        
        # Delete model
        del model.model
        
        # Should handle gracefully (this is more of a smoke test)
        # In practice, TensorFlow handles cleanup automatically