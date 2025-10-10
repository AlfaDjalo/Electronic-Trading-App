import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from typing import Dict, Any, Optional, Tuple

from models.base import ModelConfig
from models.factory import ModelFactory

class ModelTrainer:
    """
    Handles model training and evaluations.
    """
    
    def __init__(self, config: ModelConfig, verbose: bool=False):
        self.config = config
        self.verbose = verbose
        self.model = None
        self.results = {}

    def train_model(self, model_name: str, train_data: dict, input_shape: tuple, output_size: int):
        """
        Train a specific model.
        """
        # Create model
        self.model = ModelFactory.create_model(model_name, self.config)

        # Build and compile
        self.model.build(input_shape, output_size)
        self.model.compile()

        if self.verbose:
            print(f"Training {model_name} with input {input_shape} -> output {output_size}")
            if hasattr(self.model, "summary"):
                self.model.summary()
            print(train_data['x_train'].head(5))
            print(train_data['y_train'].head(5))
            print(train_data['x_val'].head(5))
            print(train_data['y_val'].head(5))

        # Train
        history = self.model.fit(
            x_train=train_data['x_train'],
            y_train=train_data['y_train'],
            x_val=train_data['x_val'],
            y_val=train_data['y_val'],
            verbose=1 if self.verbose else 0
        )

        print("History:", history)

        return history

    def evaluate_model(self, test_data: dict) -> dict:
        """
        Evaluate the trained model.
        """
        if self.model is None:
            raise ValueError("No model trained yet")
        
        # Make predictions
        y_pred = self.model.predict(test_data['x_test'])
        y_true = test_data['y_test']

        # Calculate metrics

        # Flatten if needed
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        if y_true.ndim > 1 and y_true.shape[1] == 1:
            y_true = y_true.flatten()

        metrics = {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred))
        }

        self.results = {
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'actuals': y_true.tolist()
        }

        return self.results
    