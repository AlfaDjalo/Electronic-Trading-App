import tensorflow as tf

class BaseKerasModel(tf.keras.Model):
    """ 
    Base class for all keras models.
    """
    def __init__(self):
        super().__init__()
        self.model = None
        self.history = None

    def build_model(self, window_generator):
        """ Build the specific model architecture using window_generator to determine shapes. """
        raise NotImplementedError("Subclasses must implement build_model")

    def compile_model(self, optimizer='adam', loss='mse', metrics=['mae']):
        """ Compile the model with the specified parameters. """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, window_generator, epochs=20, patience=25, verbose=1):
        """ Train the model using WindowGenerator. """
        if self.model is None:
            # Auto-build the model from window_generator
            self.build_model(window_generator)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=8,
                    min_lr=1e-6,
                    verbose=1
                )

        self.history = self.model.fit(
            window_generator.train,
            epochs=epochs,
            validation_data=window_generator.val,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        return self.history

    def predict(self, inputs):
        """ Make predictions. """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")

        return self.model(inputs).numpy()

    def __call__(self, inputs):
        """ Make the model callable. """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")

        return self.model(inputs)

    def evaluate(self, window_generator):
        """ Evaluate on test data. """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")

        # return {'loss': 0.0, 'mae': 0.0}
        return self.model.evaluate(window_generator.test, verbose=0)


class Baseline(BaseKerasModel):
    """ 
    Baseline model that predicts no change (future = current value).
    Compatible with BaseKerasModel interface.
    """
    
    def __init__(self, label_column=None):
        super().__init__()
        self.label_column = label_column
        self.label_index = None
        self.window_generator = None
        
    def build_model(self, window_generator):
        """
        Build the baseline model using window_generator properties.
        """
        self.window_generator = window_generator

        if self.label_column is not None:
            if self.label_column not in window_generator.column_indices:
                available_cols = list(window_generator.column_indices.keys())
                raise ValueError(f"Label column {self.label_column} not found.\n Available columns: {available_cols}")
            self.label_index = window_generator.column_indices[self.label_column]

        input_shape = (window_generator.input_width, len(window_generator.column_indices))

        # Create a simple lambda layer for interface compatibility
        self.model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x, input_shape=input_shape)
        ])
        
    def compile_model(self, optimizer='adam', loss='mse', metrics=['mae']):
        """Override compile - baseline doesn't need compilation."""
        pass
        
    def fit(self, window_generator, epochs=20, patience=2, verbose=1):
        """Override fit - baseline doesn't need training."""
        if self.model is None:
            self.build_model(window_generator)

        # # Set up label index if specified
        # if self.label_column is not None:
        #     if self.label_column not in window_generator.column_indices:
        #         available_cols = list(window_generator.column_indices.keys())
        #         raise ValueError(f"Label column '{self.label_column}' not found.\n Available columns: {available_cols}")
        #     self.label_index = window_generator.column_indices[self.label_column]
        
        # Create dummy history for interface compatibility
        self.history = type('History', (), {
            'history': {'loss': [0], 'val_loss': [0]},
            'epoch': [0]
        })()
        return self.history
    
    def call(self, inputs):
        """Make predictions using baseline logic."""
        if self.window_generator is None:
            raise ValueError("Model not fitted. Call fit() first to set window_generator.")
            
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
    
    def __call__(self, inputs):
        """TensorFlow's __call__ method - delegates to call."""
        return self.call(inputs)
        
    def predict(self, inputs):
        """Make predictions."""
        if hasattr(inputs, 'element_spec'):
            predictions = []
            for batch in inputs:
                if isinstance(batch, tuple):
                    batch_inputs = batch[0]
                else:
                    batch_inputs = batch
                batch_pred = self.call(batch_inputs)
                predictions.append(batch_pred)
            result = tf.concat(predictions, axis=0)
        else:
            result = self.call(inputs)

        return result.numpy()

    def evaluate(self, window_generator):
        """ Evaluate on test data. """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")

        return {'loss': 0.0, 'mae': 0.0}


class KerasLinearRegression(BaseKerasModel):
    """ Simple linear regression model. """

    def build_model(self, window_generator):
        """ Build model automatically from window_generator properties. """
        input_shape = (window_generator.input_width, len(window_generator.column_indices))

        num_labels = len(window_generator.label_columns) if window_generator.label_columns else 1
        output_size = window_generator.label_width * num_labels

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(output_size, name='linear_output')
        ])


class KerasMLP(BaseKerasModel):
    """Multi-layer perceptron (feedforward neural network)."""
    
    def __init__(self, hidden_units=[64, 32], **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
    
    def build_model(self, window_generator):
        input_shape = (window_generator.input_width, len(window_generator.column_indices))

        num_labels = len(window_generator.label_columns) if window_generator.label_columns else 1
        output_size = window_generator.label_width * num_labels

        layers = [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten()
        ]

        for units in self.hidden_units:
            layers.extend([
                tf.keras.layers.Dense(units, activation='relu'),
                tf.keras.layers.Dropout(0.2)
            ])
        
        layers.append(tf.keras.layers.Dense(output_size, name='output'))
        self.model = tf.keras.Sequential(layers)


class KerasLSTM(BaseKerasModel):
    """LSTM model for time series."""
    
    def __init__(self, lstm_units=64, dropout=0.2, activation='tanh', 
                    l1_reg=0.0, seed=0, loss='mean_squared_error', 
                    optimizer='adam', use_batch_norm=True, use_layer_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.activation = activation
        self.l1_reg = l1_reg
        self.seed = seed
        self.loss = loss
        self.optimizer = optimizer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

    def build_model(self, window_generator):
        input_shape = (window_generator.input_width, len(window_generator.column_indices))

        num_labels = len(window_generator.label_columns) if window_generator.label_columns else 1
        output_size = window_generator.label_width * num_labels

        layers = []
        
        # Input normalization (optional - good for time series)
        layers.append(tf.keras.layers.BatchNormalization(name='input_norm'))
        
        tf.keras.layers.Input(shape=input_shape),

        # LSTM layer
        layers.append(tf.keras.layers.LSTM(
            self.lstm_units, 
            activation=self.activation,
            dropout=self.dropout,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            bias_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            recurrent_initializer=tf.keras.initializers.Orthogonal(seed=self.seed),
            kernel_regularizer=tf.keras.regularizers.l1(self.l1_reg),
            unroll=True,
            return_sequences=False  # Set to True if you want normalization after LSTM
        ))
        
        # Post-LSTM normalization
        if self.use_batch_norm:
            layers.append(tf.keras.layers.BatchNormalization(name='lstm_batch_norm'))
        elif self.use_layer_norm:
            layers.append(tf.keras.layers.LayerNormalization(name='lstm_layer_norm'))
        
        # Optional: Additional dropout after normalization
        if self.dropout > 0:
            layers.append(tf.keras.layers.Dropout(self.dropout, name='post_norm_dropout'))
        
        # Dense output layer
        layers.append(tf.keras.layers.Dense(
            output_size, 
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            bias_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            kernel_regularizer=tf.keras.regularizers.l1(self.l1_reg),
            name='lstm_output'
        ))

        self.model = tf.keras.Sequential(layers)

        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.LSTM(self.lstm_units, 
        #     activation = self.activation,
        #     dropout=self.dropout, 
        #     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
        #     bias_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
        #     recurrent_initializer=tf.keras.initializers.Orthogonal(seed=self.seed),
        #     kernel_regularizer=tf.keras.regularizers.l1(self.l1_reg),
        #     input_shape=input_shape,
        #     unroll=True),
        #     tf.keras.layers.Dense(
        #         output_size, 
        #         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
        #         bias_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
        #         kernel_regularizer=tf.keras.regularizers.l1(self.l1_reg),
        #         name='lstm_output'
        #     )        ])


class KerasCNN(BaseKerasModel):
    """1D CNN for time series."""
    
    def __init__(self, filters=64, kernel_size=3, dense_layer=50, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dense_layer = dense_layer

    def build_model(self, window_generator):
        input_shape = (window_generator.input_width, len(window_generator.column_indices))

        num_labels = len(window_generator.label_columns) if window_generator.label_columns else 1
        output_size = window_generator.label_width * num_labels

        self.model = tf.keras.Sequential([

            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                                activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(self.dense_layer, activation='relu'),
            tf.keras.layers.Dense(output_size, name='cnn_output')
        ])

# Model factory for easy instantiation
def create_model(model_type, **kwargs):
    """
    Factory function to create models by name.
    
    Args:
        model_type (str): Name of the model ('linear', 'mlp', 'lstm', 'cnn')
        **kwargs: Arguments to pass to the model constructor
    
    Returns:
        Model instance
    """
    models = {
        'linear': LinearRegression,
        'mlp': MultiLayerPerceptron,
        'lstm': LSTM,
        'cnn': CNN
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)