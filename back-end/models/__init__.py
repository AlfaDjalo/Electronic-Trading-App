# models/__init__.py

"""
Models package for time series prediction.

This package contains all model implementations including:
- Keras-based models (Linear, LSTM, CNN, MLP)
- Custom models (Baseline)
- Model factory functions
"""

# Import all model classes for easy access
from .base import BaseModel, ModelConfig
# from .keras_models import (
#     BaseKerasModel,
#     Baseline,
#     KerasLinearRegression,
#     KerasLSTM,
#     KerasCNN,
#     KerasMLP,
#     # MultiLayerPerceptron,
#     create_model
# )

# from .baseline import Baseline

# Define what gets imported with "from models import *"
__all__ = [
    'BaseModel',
    'ModelConfig',
    # 'BaseKerasModel',
    # 'Baseline',
    # 'KerasLinearRegression', 
    # 'KerasLSTM',
    # 'KerasCNN',
    # 'KerasMLP',
    # # 'MultiLayerPerceptron',
    # # 'Baseline',
    # 'create_model'
]

# Optional: Create convenient aliases
# Linear = LinearRegression
# MLP = MultiLayerPerceptron

# # Optional: Version info
# __version__ = '1.0.0'