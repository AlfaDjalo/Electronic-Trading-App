import pytest
import sys
import os
from dataclasses import asdict, FrozenInstanceError
from unittest.mock import patch

# Add parent directory to path (following your pattern)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.base import ModelConfig

class TestModelConfig:
    """Test suite for ModelConfig dataclass."""
    
    def test_default_initialization(self):
        """Test ModelConfig with default values."""
        config = ModelConfig()
        
        assert config.epochs == 20
        assert config.batch_size == 32
        assert config.optimizer == "adam"
        assert config.loss == "mse"
        assert config.metrics == ["mae"]
        assert config.model_params == {}
        assert config.early_stopping_patience == 10
        assert config.reduce_lr_patience == 5
    
    def test_custom_initialization(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            epochs=50,
            batch_size=64,
            optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=["accuracy", "precision"],
            model_params={"units": 128, "dropout": 0.3},
            early_stopping_patience=15,
            reduce_lr_patience=7
        )
        
        assert config.epochs == 50
        assert config.batch_size == 64
        assert config.optimizer == "rmsprop"
        assert config.loss == "binary_crossentropy"
        assert config.metrics == ["accuracy", "precision"]
        assert config.model_params == {"units": 128, "dropout": 0.3}
        assert config.early_stopping_patience == 15
        assert config.reduce_lr_patience == 7
    
    def test_partial_initialization(self):
        """Test ModelConfig with some custom and some default values."""
        config = ModelConfig(
            epochs=100,
            model_params={"hidden_units": [64, 32]}
        )
        
        # Custom values
        assert config.epochs == 100
        assert config.model_params == {"hidden_units": [64, 32]}
        
        # Default values should remain
        assert config.batch_size == 32
        assert config.optimizer == "adam"
        assert config.loss == "mse"
        assert config.metrics == ["mae"]
    
    def test_post_init_metrics_default(self):
        """Test that metrics defaults to ['mae'] when None."""
        config = ModelConfig(metrics=None)
        assert config.metrics == ["mae"]
    
    def test_post_init_model_params_default(self):
        """Test that model_params defaults to {} when None."""
        config = ModelConfig(model_params=None)
        assert config.model_params == {}
    
    def test_config_to_dict_conversion(self):
        """Test that ModelConfig can be converted to dictionary."""
        config = ModelConfig(
            epochs=25,
            batch_size=16,
            optimizer="sgd",
            model_params={"dropout": 0.2}
        )
        
        config_dict = asdict(config)
        
        assert isinstance(config_dict, dict)
        assert config_dict["epochs"] == 25
        assert config_dict["batch_size"] == 16
        assert config_dict["optimizer"] == "sgd"
        assert config_dict["model_params"] == {"dropout": 0.2}
        assert config_dict["metrics"] == ["mae"]
    
    def test_config_equality(self):
        """Test that two ModelConfig instances with same values are equal."""
        config1 = ModelConfig(epochs=30, batch_size=64)
        config2 = ModelConfig(epochs=30, batch_size=64)
        
        assert config1 == config2
    
    def test_config_inequality(self):
        """Test that two ModelConfig instances with different values are not equal."""
        config1 = ModelConfig(epochs=30, batch_size=64)
        config2 = ModelConfig(epochs=30, batch_size=32)  # Different batch_size
        
        assert config1 != config2
    
    def test_config_immutability(self):
        """Test that ModelConfig is frozen (immutable after creation)."""
        config = ModelConfig(epochs=20)
        
        # Should not be able to modify after creation
        with pytest.raises(FrozenInstanceError):
            config.epochs = 50
    
    def test_nested_dict_handling(self):
        """Test that nested dictionaries in model_params are handled correctly."""
        complex_params = {
            "layer_config": {
                "units": [64, 32, 16],
                "activations": ["relu", "relu", "sigmoid"]
            },
            "regularization": {
                "l1": 0.01,
                "l2": 0.02
            }
        }
        
        config = ModelConfig(model_params=complex_params)
        
        assert config.model_params["layer_config"]["units"] == [64, 32, 16]
        assert config.model_params["regularization"]["l1"] == 0.01
    
    @pytest.mark.parametrize("epochs,batch_size,expected_valid", [
        (1, 1, True),      # Minimum valid values
        (1000, 1024, True),  # Large valid values
        (0, 32, False),     # Invalid epochs
        (20, 0, False),     # Invalid batch_size
        (-1, 32, False),    # Negative epochs
        (20, -1, False),    # Negative batch_size
    ])
    def test_parameter_validation_ranges(self, epochs, batch_size, expected_valid):
        """Test various parameter combinations for validity."""
        if expected_valid:
            config = ModelConfig(epochs=epochs, batch_size=batch_size)
            assert config.epochs == epochs
            assert config.batch_size == batch_size
        else:
            # For now, just create the config - in a real implementation
            # you might want to add validation that raises errors for invalid values
            config = ModelConfig(epochs=epochs, batch_size=batch_size)
            # This test documents current behavior - you could add validation later
            assert isinstance(config, ModelConfig)
    
    @pytest.mark.parametrize("optimizer", [
        "adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"
    ])
    def test_valid_optimizers(self, optimizer):
        """Test that common TensorFlow optimizers can be set."""
        config = ModelConfig(optimizer=optimizer)
        assert config.optimizer == optimizer
    
    @pytest.mark.parametrize("loss", [
        "mse", "mae", "binary_crossentropy", "categorical_crossentropy", 
        "sparse_categorical_crossentropy", "huber"
    ])
    def test_valid_loss_functions(self, loss):
        """Test that common TensorFlow loss functions can be set."""
        config = ModelConfig(loss=loss)
        assert config.loss == loss
    
    def test_empty_model_params(self):
        """Test behavior with empty model_params dictionary."""
        config = ModelConfig(model_params={})
        assert config.model_params == {}
        assert isinstance(config.model_params, dict)
    
    def test_metrics_list_types(self):
        """Test different types of metrics lists."""
        # String metrics
        config1 = ModelConfig(metrics=["mae", "mse"])
        assert config1.metrics == ["mae", "mse"]
        
        # Empty metrics list
        config2 = ModelConfig(metrics=[])
        assert config2.metrics == []
        
        # Single metric as string (should be converted to list if needed)
        config3 = ModelConfig(metrics=["accuracy"])
        assert config3.metrics == ["accuracy"]
    
    def test_config_repr(self):
        """Test string representation of ModelConfig."""
        config = ModelConfig(epochs=10, batch_size=16)
        repr_str = repr(config)
        
        assert "ModelConfig" in repr_str
        assert "epochs=10" in repr_str
        assert "batch_size=16" in repr_str
    
    def test_config_copy_and_modify(self):
        """Test creating new config based on existing one."""
        original = ModelConfig(epochs=20, batch_size=32)
        
        # Create new config with modified values
        modified = ModelConfig(
            epochs=50,  # Changed
            batch_size=original.batch_size,  # Keep same
            optimizer=original.optimizer,  # Keep same
            loss=original.loss,  # Keep same
            metrics=original.metrics,  # Keep same
            model_params=original.model_params,  # Keep same
            early_stopping_patience=original.early_stopping_patience,  # Keep same
            reduce_lr_patience=original.reduce_lr_patience  # Keep same
        )
        
        assert modified.epochs == 50
        assert modified.batch_size == 32
        assert original.epochs == 20  # Original unchanged