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


class GatherLayer(tf.keras.layers.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = indices

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=-1)
    
    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], len(self.indices))


class LOBCNNModel(BaseModel):
    """
    Expects windowed input shape: (batch, input_width, n_features.)
    Reshapes features at each timestep into (levels*2, features_per_level),
    applies Conv1D across levels, then Conv1D across time, flattens and outputs.
    ModelConfig.model_params supports:
    - levels (int): number of price levels per side (default 5)
    - features_per_level (int): usually 2 (price, volume) (default 2)
    - conv_filters (int): Conv1D filters (default 8)
    - kernel_size (int): Conv1D kernel size (default 1)
    - dense_units (int): Dense hidden units after conv (default 64)    
    """
    def build_architecture(self, input_shape: Tuple[int, ...], output_size: int) -> tf.keras.Model:
        params = getattr(self.config, "model_params", {}) or {}
        features_per_level = int(params.get("features_per_level", 2))
        conv_filters = int(params.get("conv_filters", 8))
        kernel_size = int(params.get("kernel_size", 1))
        dense_units = int(params.get("dense_units", 64))

        # input_shape: (input_width, n_features)
        inputs = tf.keras.layers.Input(shape=input_shape, name="lob_input")

        input_width = input_shape[0]
        n_features = input_shape[-1]
        block = features_per_level * 2

        # If feature count not divisible by expected block size, attempt a safe trim.
        if n_features % block != 0:
            remainder = n_features % block
            # If remainder is small (e.g. 1) we assume an extra trailing column (target)
            # and drop the last `remainder` channels. If remainder is large, it's likely
            # the feature layout is wrong and we raise an error below.
            if remainder <= 3:
                # Trim the last `remainder` columns across the feature axis
                tf.print(f"LOBCNNModel: trimming last {remainder} feature column(s) to match expected LOB layout.")
                x = tf.keras.layers.Lambda(lambda t: t[..., :-remainder], name="lob_trim_extra_features")(inputs)
                n_features = n_features - remainder
            else:
                raise ValueError(
                    f"Number of features ({input_shape[-1]}) is not compatible with features_per_level*2 ({block}). "
                    f"Remaining columns after dividing by block: {remainder}. Expected format: "
                    f"[bid_price_0..N, bid_vol_0..N, ask_price_0..N, ask_vol_0..N]"
                )
        else:
            x = inputs

        # Recompute levels after possible trimming
        if n_features % block != 0:
            # defensive check
            raise ValueError(
                f"After trimming, number of features ({n_features}) must be divisible by features_per_level*2 ({block})."
            )

        levels = n_features // (features_per_level * 2)
        expected = levels * features_per_level * 2  # bid+ask

        if levels <= 0:
            raise ValueError(f"Insufficient features ({n_features}) to build LOB model with features_per_level={features_per_level}.")

        # Build LOB feature grouping assuming ordering:
        # [bid_price_0..levels-1, bid_vol_0..levels-1, ask_price_0..levels-1, ask_vol_0..levels-1]
        price_bid = GatherLayer(indices=range(0, levels), name="gather_price_bid")(x)
        vol_bid = GatherLayer(indices=range(levels, 2*levels), name="gather_vol_bid")(x)
        price_ask = GatherLayer(indices=range(2*levels, 3*levels), name="gather_price_ask")(x)
        vol_ask = GatherLayer(indices=range(3*levels, 4*levels), name="gather_vol_ask")(x)

        # Stack bid features
        bid = tf.keras.layers.Lambda(
            lambda inputs: tf.stack(inputs, axis=-1),
            name="stack_bid"
        )([price_bid, vol_bid])

        ask = tf.keras.layers.Lambda(
            lambda inputs: tf.stack(inputs, axis=-1),
            name="stack_ask"
        )([price_ask, vol_ask])

        x = tf.keras.layers.Concatenate(axis=2, name="lob_bid_ask_concat")([bid, ask])

        x = tf.keras.layers.Reshape((input_width * levels * 2, features_per_level), name="merge_time_levels")(x)

        # Conv1D expected shape (batch, steps, channels)
        x = tf.keras.layers.Conv1D(filters=conv_filters, kernel_size=kernel_size, activation="relu", name="lob_conv1")(x)
        x = tf.keras.layers.Flatten(name="lob_flatten")(x)
        x = tf.keras.layers.Dense(dense_units, activation="relu", name="lob_dense")(x)
        outputs = tf.keras.layers.Dense(output_size, activation="linear", name="lob_output")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="LOBCNNModel")
        return model

