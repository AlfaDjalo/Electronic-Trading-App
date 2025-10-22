import tensorflow as tf
import numpy as np
from typing import Dict, Any, Optional, Tuple

from .base import ModelConfig, BaseModel

import tensorflow as tf
import numpy as np
from typing import Tuple

from models.base import BaseModel  # assuming your BaseModel provides build(), compile(), etc.

class BaselineModel(BaseModel):
    """
    Baseline model: predicts that y_{t+k} = y_t (persistence model).
    Works with any forecast horizon (shift) defined in the WindowGenerator.
    """
    def build_architecture(self, input_shape: Tuple[int, ...], output_size: int) -> tf.keras.Model:
        # Same as LinearModel: flatten then dense
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(output_size, name='baseline_output')
        ])

        # Build once so weights exist
        model.build((None, *input_shape))

        # Get the Dense layer and overwrite its weights
        dense_layer = model.get_layer('baseline_output')

        # Create kernel and bias arrays with the right shapes
        kernel_shape = dense_layer.kernel.shape  # e.g. (input_width * n_features, output_size)
        bias_shape = dense_layer.bias.shape      # e.g. (output_size,)

        # Initialize kernel as all zeros, but set last feature → 1.0
        # (since you want to copy the last observed value)
        kernel = np.zeros(kernel_shape, dtype=np.float32)
        kernel[-1, :] = 1.0  # the last element in flattened input maps directly to output

        bias = np.zeros(bias_shape, dtype=np.float32)

        dense_layer.set_weights([kernel, bias])

        # Freeze the layer to avoid training
        dense_layer.trainable = False
        model.trainable = False

        return model

class BaselineModel_old2(BaseModel):
    """
    Baseline model: predicts the last observed value repeated over the forecast horizon.
    """
    def __init__(self, name=None, config=None, **kwargs):
        super().__init__(name=name, config=config, **kwargs)
        self.output_size = None
        self.model = None      

    def build_architecture(self, input_shape, output_size):
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Last timestep, last feature: (batch, input_width, features) -> (batch,)
        last_value = tf.keras.layers.Lambda(
            lambda x: x[:, -1, -1]
        )(inputs)
        
        # Reshape to (batch, 1) then tile to (batch, output_size)
        last_value = tf.keras.layers.Reshape((1,))(last_value)
        
        if output_size > 1:
            outputs = tf.keras.layers.Lambda(
                lambda x: tf.tile(x, [1, output_size])
            )(last_value)
        else:
            outputs = last_value
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def build_architecture_old(self, input_shape, output_size):
        self.input_shape = input_shape
        self.output_size = output_size

        inputs = tf.keras.layers.Input(shape=input_shape)
        # Take the last timestep from the input (shape: batch, n_features)
        last_step = tf.keras.layers.Lambda(lambda x: x[:, -1, :], name="last_step")(inputs)
        # Optionally select the first feature if you have multiple inputs
        if input_shape[-1] > 1:
            last_value = tf.keras.layers.Lambda(lambda x: x[:, 0:1], name="select_first_feature")(last_step)
        else:
            last_value = last_step

        outputs = last_value  # no Dense layer, no learning
        return tf.keras.Model(inputs, outputs, name="BaselineModel")


    # def build_architecture(self, input_shape, output_size):
    #     """Build model that outputs last timestep repeated."""
        # self.input_shape = input_shape
        # self.output_size = output_size

        # inputs = tf.keras.layers.Input(shape=input_shape)
        
        # # Take last timestep, extract single feature (target column is last)
        # last_value = tf.keras.layers.Lambda(lambda x: x[:, -1, -1:])(inputs)
        
        # # Repeat for forecast_period steps
        # # Shape: (batch, 1) -> (batch, output_size)
        # outputs = tf.keras.layers.Lambda(
        #     lambda x: tf.tile(x, [1, output_size])
        # )(last_value)
        
        # return tf.keras.Model(inputs=inputs, outputs=outputs)

    def fit(self, x_train=None, y_train=None, x_val=None, y_val=None, *args, **kwargs):
        """
        No training needed.
        Just build architecture if not built yet.
        """
        if self.model is None:
            input_shape = x_train.shape[1:]
            output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
            self.build_architecture(input_shape, output_size)
        return self

    def predict(self, X, verbose=0):
        """
        Return 2D predictions (batch, n_targets). Use the underlying keras model
        for consistency with other models.
        """
        return self.model(X, training=False).numpy()


    def predict_also_old_not_giving_correct_output(self, X, *args, **kwargs):
        print("X.shape")
        print(X.shape)

        if X.ndim == 3:
            last_obs = X[:, -1, :]
        elif X.ndim == 2:
            last_obs = X
 
        # Expand dimension safely → (batch, 1, features)
        last_obs_expanded = np.expand_dims(last_obs, axis=1)

        # Repeat along the forecast horizon axis → (batch, output_size, features)
        preds = np.repeat(last_obs_expanded, self.output_size, axis=1)

        return preds
        # X shape: [batch, input_width, features]
        # last_obs = X[:, -1, :]  # last observed value in the input window
        # print("Input (first 5 rows):")
        # print(X[:5])
        # print("Output (first 5 rows):")
        # print(np.repeat(last_obs[:, np.newaxis, :], self.output_size, axis=1)[:5])
        # return np.repeat(last_obs[:, np.newaxis, :], self.output_size, axis=1)
    
    def predict_old(self, X, *args, **kwargs):
        """
        Predict last observed value repeated across forecast horizon.
        """
        last_step = X[:, -1, :]  # shape [batch, features]
        repeated = np.repeat(last_step[:, np.newaxis, :], self.output_size, axis=1)

        # Optional debug
        print("Input (first 5 rows):")
        print(X[:5])
        print("Output (first 5 rows):")
        print(repeated[:5])

        return repeated    

class BaselineModel_old(BaseModel):
    """ 
    Baseline model that predicts the last known value.
    """
    # def build_architecture(self, input_shape: Tuple[int, ...], output_size: int) -> tf.keras.Model:
    #     print("Using baseline model.")
    #     inputs = tf.keras.layers.Input(shape=input_shape)
    #     # last_step = tf.keras.layers.Lambda(lambda x: x[:, -1 :])(inputs)
    #     last_step = tf.keras.layers.Lambda(lambda x: x[:, -1])(inputs)
    #     outputs = tf.keras.layers.Dense(output_size, activation="linear")(last_step)
    #     return tf.keras.Model(inputs=inputs, outputs=outputs)

    def build_architecture(self, input_shape: Tuple[int, ...], output_size: int) -> tf.keras.Model:
        # We don’t really need a learnable TF model here,
        # but we wrap it in a Lambda layer so it looks consistent.
        horizon = self.config.model_params.get("inputWidth", 1)

        inputs = tf.keras.layers.Input(shape=input_shape)

        def lag_fn(x):
            """
            x: (batch, input_width, features)
            output: (batch, label_width, features)
            """
            # Grab the first timestep in the input window, which is lagged by input_width
            lagged_series = x[:, 0:1, :]  # shape: (batch, 1, features)

            # Repeat it along label_width to match output shape
            repeated = tf.repeat(lagged_series, repeats=label_width, axis=1)
            return repeated

        lagged = tf.keras.layers.Lambda(lag_fn)(inputs)

        # Dense layer to ensure correct output shape
        outputs = tf.keras.layers.Dense(output_size, activation="linear")(lagged)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def fit(self, x_train=None, y_train=None, x_val=None, y_val=None, *args, **kwargs):
        """
        No training required; just build the model for predict().
        Accepts arbitrary args/kwargs to match pipeline signature.
        """
        if not hasattr(self, "model") or self.model is None:
            input_shape = x_train.shape[1:]
            output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
            self.model = self.build_architecture(input_shape, output_size)
        return self

    # def fit(self, X, y, *args, **kwargs):
    #     # Override training to do nothing
    #     self.model = self.build_architecture(X.shape[1:], y.shape[1])
    #     return self

    def predict(self, X, *args, **kwargs):
        # Use TF model for consistency
        print("Model Input:")
        print(X[:5])
        print("Model Output:")
        print(self.model(X, training=False).numpy()[:5])
        return self.model(X, training=False).numpy()

class LinearModel(BaseModel):
    """
    Linear regression model.
    """
    def build_architecture(self, input_shape: Tuple[int, ...], output_size: int) -> tf.keras.Model:
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(output_size, name='linear_output')
        ])
    
class MLPModel(BaseModel):
    """
    Multi-layer perceptron model.
    """
    def build_architecture(self, input_shape: Tuple[int, ...], output_size: int) -> tf.keras.Model:
        hidden_units = self.config.model_params.get('hidden_units', [64, 32])
        dropout_rate = self.config.model_params.get('dropout_rate', 0.2)

        layers = [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten()
        ]

        for units in hidden_units:
            layers.extend([
                tf.keras.layers.Dense(units, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate)
            ])

        layers.append(tf.keras.layers.Dense(output_size, name='output'))
        return tf.keras.Sequential(layers)

class LSTMModel(BaseModel):
    """
    LSTM model for time series.
    """
    def build_architecture(self, input_shape: Tuple[int, ...], output_size: int) -> tf.keras.Model:
        lstm_units = self.config.model_params.get('lstm_units', 64)
        dropout = self.config.model_params.get('dropout', 0.2)
        use_layer_norm = self.config.model_params.get('use_layer_norm', True)
        use_batch_norm = self.config.model_params.get('use_batch_norm', True)

        layers = [tf.keras.layers.Input(shape=input_shape)]

        if use_batch_norm:
            layers.append(tf.keras.layers.BatchNormalization())
        if use_layer_norm:
            layers.append(tf.keras.layers.LayerNormalization())

        layers.extend([
            tf.keras.layers.LSTM(lstm_units, dropout=dropout),
            tf.keras.layers.Dense(output_size, name='lstm_output')
        ])

        return tf.keras.Sequential(layers)
