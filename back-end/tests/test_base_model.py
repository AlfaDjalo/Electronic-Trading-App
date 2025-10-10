import pytest
import sys
import os
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock
from abc import ABC

# Add parent directory to path (following your pattern)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.base import ModelConfig, BaseModel

# =============================================================================
# TEST FIXTURES (following your pattern)
# =============================================================================

@pytest.fixture
def basic_config():
    """Basic ModelConfig for testing."""
    return ModelConfig(
        epochs=2,  # Small for fast testing
        batch_size=16,
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
        early_stopping_patience=5,
        reduce_lr_patience=3
    )

@pytest.fixture
def advanced_config():
    """Advanced ModelConfig with custom parameters."""
    return ModelConfig(
        epochs=10,
        batch_size=32,
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy", "precision"],
        model_params={"units": 64, "dropout": 0.3},
        early_stopping_patience=15,
        reduce_lr_patience=7
    )

@pytest.fixture
def sample_input_shapes():
    """Various input shapes for testing."""
    return [
        (10, 3),    # 10 timesteps, 3 features
        (20, 1),    # 20 timesteps, 1 feature  
        (5, 10),    # 5 timesteps, 10 features
        (1, 1),     # Minimal case
    ]

@pytest.fixture
def sample_output_sizes():
    """Various output sizes for testing."""
    return [1, 2, 5, 10]

# =============================================================================
# CONCRETE TEST IMPLEMENTATION OF BaseModel
# =============================================================================

class TestableModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""
    
    def build_architecture(self, input_shape, output_size):
        """Simple implementation for testing."""
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(output_size, name='test_output')
        ])

class TestableModelWithError(BaseModel):
    """Model that raises error during build for testing error handling."""
    
    def build_architecture(self, input_shape, output_size):
        """Intentionally raises error for testing."""
        raise ValueError("Test error during build")

# =============================================================================
# TEST CLASS
# =============================================================================

class TestBaseModel:
    """Test suite for BaseModel abstract class."""
    
    def test_base_model_is_abstract(self):
        """Test that BaseModel cannot be instantiated directly."""
        config = ModelConfig()
        
        with pytest.raises(TypeError):
            BaseModel("test", config)
    
    def test_concrete_model_instantiation(self, basic_config):
        """Test that concrete implementation can be instantiated."""
        model = TestableModel("test_model", basic_config)
        
        assert model.name == "test_model"
        assert model.config == basic_config
        assert model.model is None
        assert model.history is None
        assert not model.is_built
        assert not model.is_compiled
    
    def test_initial_state(self, basic_config):
        """Test initial state of new model."""
        model = TestableModel("test", basic_config)
        
        # Check all initial values
        assert model.name == "test"
        assert model.config is basic_config
        assert model.model is None
        assert model.history is None
        assert model.is_built is False
        assert model.is_compiled is False
    
    @pytest.mark.parametrize("input_shape,output_size", [
        ((10, 3), 1),
        ((20, 5), 2),
        ((5, 1), 10),
    ])
    def test_model_building(self, basic_config, input_shape, output_size):
        """Test model building with various shapes."""
        model = TestableModel("test", basic_config)
        
        # Initially not built
        assert not model.is_built
        assert model.model is None
        
        # Build model
        model.build(input_shape, output_size)
        
        # Should now be built
        assert model.is_built
        assert model.model is not None
        assert isinstance(model.model, tf.keras.Model)
        
        # Check input/output shapes
        assert model.model.input_shape[1:] == input_shape
        assert model.model.output_shape[1] == output_size
    
    def test_model_building_idempotent(self, basic_config):
        """Test that calling build() multiple times doesn't rebuild."""
        model = TestableModel("test", basic_config)
        
        # Build once
        model.build((10, 3), 1)
        first_model_id = id(model.model)
        
        # Build again - should not rebuild
        model.build((20, 5), 2)  # Different parameters
        second_model_id = id(model.model)
        
        # Should be same model object
        assert first_model_id == second_model_id
        # Should maintain original shape
        assert model.model.input_shape[1:] == (10, 3)
    
    def test_model_compilation_requires_build(self, basic_config):
        """Test that compilation requires model to be built first."""
        model = TestableModel("test", basic_config)
        
        with pytest.raises(ValueError, match="Model must be built before compilation"):
            model.compile()
    
    def test_model_compilation(self, basic_config):
        """Test model compilation after building."""
        model = TestableModel("test", basic_config)
        
        # Build first
        model.build((10, 3), 1)
        assert not model.is_compiled
        
        # Then compile
        model.compile()
        
        assert model.is_compiled
        assert model.model.optimizer is not None
        assert model.model.loss is not None
    
    def test_model_compilation_idempotent(self, basic_config):
        """Test that calling compile() multiple times doesn't recompile."""
        model = TestableModel("test", basic_config)
        model.build((10, 3), 1)
        
        # Compile once
        model.compile()
        first_optimizer = model.model.optimizer
        
        # Compile again
        model.compile()
        second_optimizer = model.model.optimizer
        
        # Should be same optimizer
        assert first_optimizer is second_optimizer
    
    def test_get_callbacks_default(self, basic_config):
        """Test default callbacks generation."""
        model = TestableModel("test", basic_config)
        callbacks = model.get_callbacks()
        
        assert isinstance(callbacks, list)
        assert len(callbacks) == 2  # EarlyStopping + ReduceLROnPlateau
        
        # Check callback types
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert "EarlyStopping" in callback_types
        assert "ReduceLROnPlateau" in callback_types
    
    def test_get_callbacks_disabled(self):
        """Test callbacks when patience is 0 (disabled)."""
        config = ModelConfig(early_stopping_patience=0, reduce_lr_patience=0)
        model = TestableModel("test", config)
        callbacks = model.get_callbacks()
        
        assert isinstance(callbacks, list)
        assert len(callbacks) == 0
    
    def test_get_callbacks_partial(self):
        """Test callbacks when only some are enabled."""
        config = ModelConfig(early_stopping_patience=10, reduce_lr_patience=0)
        model = TestableModel("test", config)
        callbacks = model.get_callbacks()
        
        assert len(callbacks) == 1
        assert type(callbacks[0]).__name__ == "EarlyStopping"
    
    def test_fit_requires_compilation(self, basic_config):
        """Test that fit() requires model to be compiled."""
        model = TestableModel("test", basic_config)
        model.build((10, 3), 1)
        
        # Sample data
        x_train = np.random.randn(32, 10, 3)
        y_train = np.random.randn(32, 1)
        
        with pytest.raises(ValueError, match="Model must be compiled before training"):
            model.fit(x_train, y_train)
    
    def test_fit_basic(self, basic_config):
        """Test basic model fitting."""
        model = TestableModel("test", basic_config)
        model.build((10, 3), 1)
        model.compile()
        
        # Sample data
        x_train = np.random.randn(32, 10, 3)
        y_train = np.random.randn(32, 1)
        
        # Fit model
        history = model.fit(x_train, y_train, verbose=0)
        
        assert model.history is not None
        assert model.history is history
        assert hasattr(history, 'history')
        assert 'loss' in history.history
    
    def test_fit_with_validation(self, basic_config):
        """Test model fitting with validation data."""
        model = TestableModel("test", basic_config)
        model.build((10, 3), 1)
        model.compile()
        
        # Sample data
        x_train = np.random.randn(32, 10, 3)
        y_train = np.random.randn(32, 1)
        x_val = np.random.randn(16, 10, 3)
        y_val = np.random.randn(16, 1)
        
        # Fit model
        history = model.fit(x_train, y_train, x_val, y_val, verbose=0)
        
        assert 'loss' in history.history
        assert 'val_loss' in history.history
    
    def test_predict_requires_build(self, basic_config):
        """Test that predict() requires model to be built."""
        model = TestableModel("test", basic_config)
        
        x = np.random.randn(5, 10, 3)
        
        with pytest.raises(ValueError, match="Model must be built before prediction"):
            model.predict(x)
    
    def test_predict_basic(self, basic_config):
        """Test basic prediction functionality."""
        model = TestableModel("test", basic_config)
        model.build((10, 3), 1)
        model.compile()
        
        # Sample input
        x = np.random.randn(5, 10, 3)
        
        # Make prediction
        predictions = model.predict(x)
        
        assert predictions.shape == (5, 1)
        assert isinstance(predictions, np.ndarray)
    
    def test_evaluate_requires_build(self, basic_config):
        """Test that evaluate() requires model to be built."""
        model = TestableModel("test", basic_config)
        
        x = np.random.randn(5, 10, 3)
        y = np.random.randn(5, 1)
        
        with pytest.raises(ValueError, match="Model must be built before evaluation"):
            model.evaluate(x, y)
    
    def test_evaluate_basic(self, basic_config):
        """Test basic evaluation functionality."""
        model = TestableModel("test", basic_config)
        model.build((10, 3), 1)
        model.compile()
        
        # Sample data
        x = np.random.randn(16, 10, 3)
        y = np.random.randn(16, 1)
        
        # Evaluate
        results = model.evaluate(x, y, verbose=0)
        
        assert isinstance(results, (list, float))
        # Should return loss and metrics
    
    def test_config_parameter_usage(self, advanced_config):
        """Test that model uses config parameters correctly."""
        model = TestableModel("test", advanced_config)
        model.build((10, 3), 1)
        model.compile()
        
        # Check optimizer
        assert model.model.optimizer.__class__.__name__.lower() == "rmsprop"
        
        # Check that model_params are accessible
        assert model.config.model_params["units"] == 64
        assert model.config.model_params["dropout"] == 0.3
    
    def test_error_handling_during_build(self, basic_config):
        """Test error handling when build_architecture fails."""
        model = TestableModelWithError("test", basic_config)
        
        with pytest.raises(ValueError, match="Test error during build"):
            model.build((10, 3), 1)
        
        # Model should remain in unbuilt state
        assert not model.is_built
        assert model.model is None
    
    @pytest.mark.parametrize("name", ["", "test", "test_model_123", "UPPERCASE"])
    def test_model_name_variants(self, basic_config, name):
        """Test model creation with various name formats."""
        model = TestableModel(name, basic_config)
        assert model.name == name
    
    def test_model_state_consistency(self, basic_config):
        """Test that model state remains consistent through operations."""
        model = TestableModel("test", basic_config)
        
        # Initial state
        assert not model.is_built and not model.is_compiled
        
        # After build
        model.build((10, 3), 1)
        assert model.is_built and not model.is_compiled
        
        # After compile
        model.compile()
        assert model.is_built and model.is_compiled
        
        # State should persist through operations
        x = np.random.randn(5, 10, 3)
        model.predict(x)
        assert model.is_built and model.is_compiled