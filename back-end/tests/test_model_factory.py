import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path (following your pattern)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.base import ModelConfig, BaseModel
from models.factory import ModelFactory
from models.implementations import LinearModel, LSTMModel, MLPModel, BaselineModel

# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def basic_config():
    """Basic ModelConfig for testing."""
    return ModelConfig(
        epochs=2,
        batch_size=16,
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

@pytest.fixture
def lstm_config():
    """Config specifically for LSTM models."""
    return ModelConfig(
        epochs=5,
        batch_size=32,
        optimizer="rmsprop",
        model_params={
            "lstm_units": 64,
            "dropout": 0.3,
            "use_batch_norm": True
        }
    )

@pytest.fixture
def mlp_config():
    """Config specifically for MLP models."""
    return ModelConfig(
        epochs=10,
        batch_size=64,
        model_params={
            "hidden_units": [128, 64, 32],
            "dropout_rate": 0.2
        }
    )

@pytest.fixture
def clean_factory():
    """Ensure factory starts with clean state for each test."""
    # Store original models
    original_models = ModelFactory._models.copy()
    
    yield ModelFactory
    
    # Restore original models after test
    ModelFactory._models = original_models

# =============================================================================
# CUSTOM TEST MODELS FOR FACTORY TESTING
# =============================================================================

class CustomTestModel(BaseModel):
    """Custom model for testing factory registration."""
    
    def build_architecture(self, input_shape, output_size):
        import tensorflow as tf
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(output_size, name='custom_output')
        ])

class BrokenTestModel(BaseModel):
    """Model with broken build method for error testing."""
    
    def build_architecture(self, input_shape, output_size):
        raise RuntimeError("Intentional error for testing")

class InvalidTestModel:
    """Class that doesn't inherit from BaseModel."""
    pass

# =============================================================================
# TEST CLASS
# =============================================================================

class TestModelFactory:
    """Test suite for ModelFactory class."""
    
    def test_factory_has_default_models(self):
        """Test that factory comes with expected default models."""
        available_models = ModelFactory.list_models()
        
        assert isinstance(available_models, list)
        assert len(available_models) > 0
        
        # Check for expected default models
        expected_models = ["baseline", "linear", "mlp", "lstm"]
        for model_name in expected_models:
            assert model_name in available_models
    
    def test_list_models_returns_correct_type(self):
        """Test that list_models returns a list."""
        models = ModelFactory.list_models()
        assert isinstance(models, list)
        assert all(isinstance(name, str) for name in models)
    
    @pytest.mark.parametrize("model_name", ["baseline", "linear", "mlp", "lstm"])
    def test_create_default_models(self, model_name, basic_config):
        """Test creating each default model type."""
        model = ModelFactory.create_model(model_name, basic_config)
        
        assert isinstance(model, BaseModel)
        assert model.name == model_name
        assert model.config == basic_config
        assert not model.is_built
        assert not model.is_compiled
    
    def test_create_linear_model(self, basic_config):
        """Test creating linear model specifically."""
        model = ModelFactory.create_model("linear", basic_config)
        
        assert isinstance(model, LinearModel)
        assert model.name == "linear"
        assert model.config is basic_config
    
    def test_create_lstm_model(self, lstm_config):
        """Test creating LSTM model with specific config."""
        model = ModelFactory.create_model("lstm", lstm_config)
        
        assert isinstance(model, LSTMModel)
        assert model.name == "lstm"
        assert model.config is lstm_config
        assert model.config.model_params["lstm_units"] == 64
    
    def test_create_mlp_model(self, mlp_config):
        """Test creating MLP model with specific config."""
        model = ModelFactory.create_model("mlp", mlp_config)
        
        assert isinstance(model, MLPModel)
        assert model.name == "mlp"
        assert model.config.model_params["hidden_units"] == [128, 64, 32]
    
    def test_create_baseline_model(self, basic_config):
        """Test creating baseline model."""
        model = ModelFactory.create_model("baseline", basic_config)
        
        assert isinstance(model, BaselineModel)
        assert model.name == "baseline"
    
    def test_create_unknown_model_raises_error(self, basic_config):
        """Test that creating unknown model raises ValueError."""
        with pytest.raises(ValueError, match=r"Unknown model type: nonexistent_model"):
            ModelFactory.create_model("nonexistent_model", basic_config)

    def test_create_model_error_includes_available_models(self, basic_config):
        """Test that error message includes list of available models."""
        try:
            ModelFactory.create_model("invalid_model", basic_config)
        except ValueError as e:
            error_msg = str(e)
            assert "Available:" in error_msg
            assert "linear" in error_msg  # Should include available models
    
    def test_register_custom_model(self, clean_factory, basic_config):
        """Test registering a new custom model."""
        # Initially not available
        assert "custom_test" not in ModelFactory.list_models()
        
        # Register custom model
        ModelFactory.register_model("custom_test", CustomTestModel)
        
        # Should now be available
        assert "custom_test" in ModelFactory.list_models()
        
        # Should be creatable
        model = ModelFactory.create_model("custom_test", basic_config)
        assert isinstance(model, CustomTestModel)
        assert model.name == "custom_test"
    
    def test_register_model_overwrites_existing(self, clean_factory, basic_config):
        """Test that registering overwrites existing model."""
        # Register custom model with same name as existing
        original_count = len(ModelFactory.list_models())
        
        ModelFactory.register_model("linear", CustomTestModel)
        
        # Count should remain the same (overwritten, not added)
        assert len(ModelFactory.list_models()) == original_count
        
        # Should create custom model, not original linear
        model = ModelFactory.create_model("linear", basic_config)
        assert isinstance(model, CustomTestModel)
    
    def test_register_invalid_model_class(self, clean_factory):
        """Test registering class that doesn't inherit from BaseModel."""
        # This should work (registration doesn't validate inheritance)
        ModelFactory.register_model("invalid", InvalidTestModel)
        
        # But creation should fail when we try to instantiate
        with pytest.raises(TypeError):
            ModelFactory.create_model("invalid", ModelConfig())
    
    def test_multiple_registrations(self, clean_factory, basic_config):
        """Test registering multiple custom models."""
        class CustomModel1(BaseModel):
            def build_architecture(self, input_shape, output_size):
                import tensorflow as tf
                return tf.keras.Sequential([
                    tf.keras.layers.Input(shape=input_shape),
                    tf.keras.layers.Dense(output_size)
                ])
        
        class CustomModel2(BaseModel):
            def build_architecture(self, input_shape, output_size):
                import tensorflow as tf
                return tf.keras.Sequential([
                    tf.keras.layers.Input(shape=input_shape),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(output_size)
                ])
        
        # Register multiple models
        ModelFactory.register_model("custom1", CustomModel1)
        ModelFactory.register_model("custom2", CustomModel2)
        
        # Both should be available
        models = ModelFactory.list_models()
        assert "custom1" in models
        assert "custom2" in models
        
        # Both should be creatable
        model1 = ModelFactory.create_model("custom1", basic_config)
        model2 = ModelFactory.create_model("custom2", basic_config)
        
        assert isinstance(model1, CustomModel1)
        assert isinstance(model2, CustomModel2)
    
    def test_factory_is_stateless(self, basic_config):
        """Test that factory doesn't maintain state between calls."""
        # Create same model type multiple times
        model1 = ModelFactory.create_model("linear", basic_config)
        model2 = ModelFactory.create_model("linear", basic_config)
        
        # Should be different instances
        assert model1 is not model2
        assert id(model1) != id(model2)
        
        # But same type and config
        assert type(model1) == type(model2)
        assert model1.config == model2.config
    
    def test_different_configs_same_model(self):
        """Test creating same model type with different configs."""
        config1 = ModelConfig(epochs=10, batch_size=32)
        config2 = ModelConfig(epochs=20, batch_size=64)
        
        model1 = ModelFactory.create_model("linear", config1)
        model2 = ModelFactory.create_model("linear", config2)
        
        assert isinstance(model1, LinearModel)
        assert isinstance(model2, LinearModel)
        assert model1.config != model2.config
        assert model1.config.epochs == 10
        assert model2.config.epochs == 20
    
    def test_case_sensitivity(self, basic_config):
        """Test that model names are case sensitive."""
        # Should work
        model = ModelFactory.create_model("linear", basic_config)
        assert isinstance(model, LinearModel)
        
        # Should fail - case sensitive
        with pytest.raises(ValueError):
            ModelFactory.create_model("LINEAR", basic_config)
        
        with pytest.raises(ValueError):
            ModelFactory.create_model("Linear", basic_config)
    
    def test_empty_model_name(self, basic_config):
        """Test behavior with empty model name."""
        with pytest.raises(ValueError):
            ModelFactory.create_model("", basic_config)
    
    def test_none_model_name(self, basic_config):
        """Test behavior with None model name."""
        with pytest.raises((ValueError, TypeError)):
            ModelFactory.create_model(None, basic_config)
    
    def test_none_config(self):
        """Test behavior with None config."""
        with pytest.raises(TypeError):
            ModelFactory.create_model("linear", None)
    
    @pytest.mark.parametrize("model_name", ["baseline", "linear", "mlp", "lstm"])
    def test_all_models_inherit_from_base(self, model_name, basic_config):
        """Test that all default models inherit from BaseModel."""
        model = ModelFactory.create_model(model_name, basic_config)
        assert isinstance(model, BaseModel)
    
    def test_model_creation_with_complex_config(self):
        """Test model creation with complex configuration."""
        complex_config = ModelConfig(
            epochs=100,
            batch_size=128,
            optimizer="adamax",
            loss="huber",
            metrics=["mae", "mse", "mape"],
            model_params={
                "lstm_units": 128,
                "dropout": 0.4,
                "use_batch_norm": False,
                "use_layer_norm": True,
                "l1_reg": 0.01,
                "seed": 42
            },
            early_stopping_patience=20,
            reduce_lr_patience=8
        )
        
        model = ModelFactory.create_model("lstm", complex_config)
        
        assert isinstance(model, LSTMModel)
        assert model.config.model_params["lstm_units"] == 128
        assert model.config.early_stopping_patience == 20
        assert model.config.metrics == ["mae", "mse", "mape"]
    
    def test_factory_thread_safety_simulation(self, basic_config):
        """Test that factory can handle concurrent-like access."""
        models = []
        
        # Simulate multiple concurrent requests
        for i in range(10):
            model = ModelFactory.create_model("linear", basic_config)
            models.append(model)
        
        # All should be valid instances
        assert len(models) == 10
        assert all(isinstance(m, LinearModel) for m in models)
        
        # All should be different instances
        model_ids = [id(m) for m in models]
        assert len(set(model_ids)) == 10  # All unique
    
    def test_register_model_with_none_name(self, clean_factory):
        """Test registering model with None name."""
        with pytest.raises((TypeError, ValueError)):
            ModelFactory.register_model(None, CustomTestModel)
    
    def test_register_model_with_empty_name(self, clean_factory):
        """Test registering model with empty name."""
        with pytest.raises(ValueError, match="Model name must be a non-empty string"):
            ModelFactory.register_model(None, CustomTestModel)
    
    def test_factory_persistence_across_calls(self, clean_factory, basic_config):
        """Test that registered models persist across multiple calls."""
        # Register a model
        ModelFactory.register_model("persistent_test", CustomTestModel)
        
        # Create model
        model1 = ModelFactory.create_model("persistent_test", basic_config)
        
        # List models
        models_list = ModelFactory.list_models()
        assert "persistent_test" in models_list
        
        # Create another instance
        model2 = ModelFactory.create_model("persistent_test", basic_config)
        
        # Both should work and be different instances
        assert isinstance(model1, CustomTestModel)
        assert isinstance(model2, CustomTestModel)
        assert model1 is not model2