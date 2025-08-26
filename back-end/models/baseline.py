import pandas as pd
import numpy as np
import tensorflow as tf

class Baseline_old(tf.keras.Model):
    def __init__(self, window_generator, label_column=None):
        """
        Baseline model that predicts no change (future = current value). 
        
        Args:
            window_generator: WindowGenerator instance.
            label_column: The name of the column to predict.
        """
        super().__init__()
        self.window_generator = window_generator
        self.label_column = label_column

        if label_column is not None:
            if label_column not in window_generator.column_indices:
                available_cols = list(window_generator.column_indices.keys())
                raise ValueError(f"Label column '{label_column}' not found.\n Available columns: {available_cols}")
            self.label_index = window_generator.column_indices[label_column]
        else:
            self.label_index = None

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        if self.label_index is None:
            # Use the last time step, all features
            last_values = inputs[:, -1, :]  # (batch_size, features)
            # Reshape to (batch_size, label_width, features)
            return tf.tile(last_values[:, tf.newaxis, :], [1, self.window_generator.label_width, 1])
        else:
            # Use the last time step of specific feature
            last_value = inputs[:, -1, self.label_index]  # (batch_size,)
            # Create output shape (batch_size, label_width, 1)
            return tf.tile(last_value[:, tf.newaxis, tf.newaxis], [1, self.window_generator.label_width, 1])            