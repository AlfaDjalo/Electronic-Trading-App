import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from typing import Dict, Any, Optional, Tuple

"""training.trainer

Utilities to train and evaluate a single model. Exposes `ModelTrainer` which
creates models via `ModelFactory`, fits them using TensorFlow datasets and
computes evaluation metrics returned to callers.
"""

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
        example_inputs, example_labels = next(iter(train_data['train']))
        input_shape = example_inputs.shape[1:]  # (input_width, n_features)
        output_size = example_labels.shape[-1]   # number of target columns

        self.model.build(input_shape, output_size)
        self.model.compile()

        if self.verbose:
            print(f"Training {model_name} with input {input_shape} -> output {output_size}")
            if hasattr(self.model, "summary"):
                self.model.summary()

        # Train
        history = self.model.model.fit(
            train_data['train'],
            validation_data=train_data['val'],
            epochs=self.config.epochs,
            verbose=1 if self.verbose else 0
        )

        return history

    def evaluate_model(self, test_data: dict) -> dict:
        """
        Evaluate the trained model.
        """
        if self.model is None:
            raise ValueError("No model trained yet")
        
        print("Evaluating model")

        y_true_list = []
        y_pred_list = []

        for x_batch, y_batch in test_data['test']:
            y_pred_batch = self.model.model.predict(x_batch)
            y_true_list.append(y_batch.numpy())
            y_pred_list.append(y_pred_batch)

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        
        # If model predicts one value per sample (persistence or models built for t+k), ensure shape (n_samples, n_targets)
        if y_pred.ndim == 3 and y_pred.shape[1] == 1:
            y_pred = y_pred[:, 0, :]
            
        # If y_true is multi-step, extract final step (t+k) for evaluation
        if y_true.ndim == 3:
            y_true_final = y_true[:, -1, :]   # final label step
        else:
            y_true_final = y_true
            
        # Flatten single-target into 1D vectors for sklearn metrics
        if y_true_final.shape[1] == 1:
            y_true_vec = y_true_final[:, 0]
            y_pred_vec = y_pred[:, 0] if y_pred.shape[1] == 1 else y_pred.flatten()
        else:
            # multivariate target -> reshape to (n_samples, n_outputs)
            y_true_vec = y_true_final.reshape(y_true_final.shape[0], -1)
            y_pred_vec = y_pred.reshape(y_pred.shape[0], -1)

        metrics = {
            'mse': float(mean_squared_error(y_true_vec, y_pred_vec)),
            'mae': float(mean_absolute_error(y_true_vec, y_pred_vec)),
            'rmse': float(np.sqrt(mean_squared_error(y_true_vec, y_pred_vec))),
            'r2': float(r2_score(y_true_vec, y_pred_vec))
        }
        
        self.results = {
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'actuals': y_true.tolist()
        }

        return self.results

    def get_base_model(self):
        """
        Return the BaseModel wrapper (includes config, history, and tf.keras.Model).
        """
        if self.model is None:
            raise ValueError("No model has been created or trained yet.")
        return self.model

    def get_model(self):
        """
        Return the underlying TensorFlow model (tf.keras.Model).
        """
        if self.model is None:
            raise ValueError("No model has been created or trained yet.")
        return self.model.model  # Access BaseModel.model

    def show_model(self, return_string=False, include_weights=False, include_values=False):
        """
        Display or return the model architecture and optionally weights and their values.
        """
        if self.model is None:
            raise ValueError("No model has been created or trained yet.")
        if not self.model.is_built:
            raise ValueError("Model must be built before showing")

        from io import StringIO
        import numpy as np

        keras_model = self.model.model  # <-- the underlying tf.keras.Model

        # Capture model summary as text
        stream = StringIO()
        keras_model.summary(print_fn=lambda x: stream.write(x + "\n"))
        summary_str = stream.getvalue()

        result = {"summary": summary_str}

        if include_weights:
            weights_info = []
            for layer in keras_model.layers:
                layer_dict = {
                    "layer": layer.name,
                    "weights": [],
                }
                for w in layer.weights:
                    weight_entry = {
                        "name": w.name,
                        "shape": tuple(w.shape.as_list())
                    }
                    if include_values:
                        w_np = w.numpy()
                        weight_entry["values"] = w_np.tolist()
                        weight_entry["mean"] = float(np.mean(w_np))
                        weight_entry["std"] = float(np.std(w_np))
                    layer_dict["weights"].append(weight_entry)
                weights_info.append(layer_dict)
            result["layers"] = weights_info

        if return_string:
            return result
        else:
            print(summary_str)
            if include_weights:
                print("\nLayer weights:")
                for layer in result["layers"]:
                    print(f"Layer: {layer['layer']}")
                    for w in layer["weights"]:
                        print(f"  {w['name']} shape={w['shape']}")
                        if include_values:
                            print(f"    mean={w['mean']:.5f}, std={w['std']:.5f}")

