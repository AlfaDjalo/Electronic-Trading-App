import tensorflow as tf
from typing import Dict, Any, Optional, Tuple
# from dataclasses import dataclass

from .base import ModelConfig, BaseModel
from .implementations import BaselineModel, LinearModel, MLPModel, LSTMModel

class ModelFactory:
    """
    Factory for creating models.
    """

    _models = {
        "baseline": BaselineModel,
        "linear": LinearModel,
        "mlp": MLPModel,
        "lstm": LSTMModel,
        # "cnn": KerasCNN,
    }

    @classmethod
    def create_model(cls, model_name: str, config: ModelConfig) -> BaseModel:
        """
        Factory function to create models by name.
        
        Args:
            name (str): Name of the model ('linear', 'mlp', 'lstm', 'cnn')
            **kwargs: Arguments to pass to the model constructor
        
        Returns:
            Model instance
        """        
        if model_name not in cls._models:
            raise ValueError(f"Unknown model type: {model_name}. Available: {list(cls._models.keys())}")
        
        if config is None:
            raise TypeError("Config cannot be None")

        model_class = cls._models[model_name]
        return model_class(name=model_name, config=config)
    
    @classmethod
    def register_model(cls, name:str, model_class: type):
        """
        Register a new model type.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Model name must be a non-empty string")
        if not isinstance(model_class, type):
            raise TypeError("Model class must be a type")

        cls._models[name] = model_class

    @classmethod
    def list_models(cls) -> list:
        """
        List available model names.
        """
        return list(cls._models.keys())
    
