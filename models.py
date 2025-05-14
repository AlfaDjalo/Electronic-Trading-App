import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import keras.initializers
from keras.models import Sequential
from keras.layers import Dense, Layer, LSTM, GRU, SimpleRNN, RNN
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Concatenate

from alphaRNN import AlphaRNN
from alphatRNN import AlphatRNN  # Import AlphaRNN and AlphatRNN modules

class ModelHandler:
    def __init__(self, data, params):
        self.data = data
        self.params = params
        self.model = None

    def regression(self):
        """Perform simple regression."""

        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(self.data['x_train'], self.data['y_train'])

        return

    # def regression_on_trend(self):
    #     """Perform regression on trend."""
    #     self.create_trend_features(3)
 
    #     print(self.df.columns)
    #     self.features = ['min_1_close', 'trend_3_day']
    #     self.target = ['close']

    #     self.prepare_features()

    #     self.model = LinearRegression(fit_intercept=True)
    #     self.model.fit(self.X_train, self.Y_train)

    #     return

    def rnn(self):
        """Perform rnn regression."""
        epochs = self.params.get('epochs', {}).get('value', 201)
        batch_size = self.params.get('batch_size', {}).get('value', 1000)
        num_units = self.params.get('num_units', {}).get('value', 10)
        l1_reg = self.params.get('l1_reg', {}).get('value', 0.0)
        seed = self.params.get('seed', {}).get('value', 0)
        activation = self.params.get('activation', {}).get('value', 'tanh')

        x_train = self.data['x_train'].values.reshape(self.data['x_train'].shape[0], self.data['x_train'].shape[1], 1)

        def SimpleRNN_():
            model = Sequential()
            model.add(SimpleRNN(num_units, activation=activation, kernel_initializer=keras.initializers.glorot_uniform(seed), \
                bias_initializer=keras.initializers.glorot_uniform(seed), recurrent_initializer=keras.initializers.orthogonal(seed), \
                kernel_regularizer=l1(l1_reg), input_shape=(x_train.shape[1], x_train.shape[-1]), unroll=True, stateful=False))  
            model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), \
                kernel_regularizer=l1(l1_reg)))
            model.compile(loss='mean_squared_error', optimizer='adam')
            return model

        es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        tf.random.set_seed(seed)
        self.model = SimpleRNN_()
        self.model.fit(x_train, self.data['y_train'], epochs=epochs, 
                  batch_size=batch_size, callbacks=[es], shuffle=False)

        return


    # def ml_regression(self, model_type="rnn", do_training=False, parameters=None):
    #     """Perform machine learning regression."""
    #     if parameters is None:
    #         parameters = {}

    #     # Extract parameters
    #     epochs = parameters.get('epochs', 2000)
    #     batch_size = parameters.get('batch_size', 1000)
    #     n_units = parameters.get('n_units', 10)
    #     l1_reg = parameters.get('l1_reg', 0.0)
    #     seed = parameters.get('seed', 0)

    #     self.create_lagged_features(4)    
    #     self.create_lagged_features(-4)
    #     print(self.df.columns)

    #     self.features = ['close', 'min_1_close', 'min_2_close', 'min_3_close', 'min_4_close']
    #     self.target = ['fut_4_close']
    #     self.prepare_features()

    #     self.standardise_input(['close'], drop=True)

    #     self.X_train = self.X_train.values.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)

    #     def SimpleRNN_():
    #         model = Sequential()
    #         model.add(SimpleRNN(n_units, activation='tanh', kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), recurrent_initializer=keras.initializers.orthogonal(seed), kernel_regularizer=l1(l1_reg), input_shape=(self.X_train.shape[1], self.X_train.shape[-1]), unroll=True, stateful=False))  
    #         model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), kernel_regularizer=l1(l1_reg)))
    #         model.compile(loss='mean_squared_error', optimizer='adam')
    #         return model

    #     def LSTM_():
    #         model = Sequential()
    #         model.add(LSTM(n_units, activation='tanh', kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), recurrent_initializer=keras.initializers.orthogonal(seed), kernel_regularizer=l1(l1_reg), input_shape=(self.X_train.shape[1], self.X_train.shape[-1]), unroll=True)) 
    #         model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), kernel_regularizer=l1(l1_reg)))
    #         model.compile(loss='mean_squared_error', optimizer='adam')
    #         return model

    #     params = {
    #         'rnn': {'function': SimpleRNN_},
    #         'lstm': {'function': LSTM_}
    #     }

    #     model_chosen = model_type
    #     es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    #     tf.random.set_seed(seed)
    #     print('Training', model_chosen, 'model')
    #     self.model = params[model_chosen]['function']()
    #     self.model.fit(self.X_train, self.Y_train, epochs=epochs, 
    #               batch_size=batch_size, callbacks=[es], shuffle=False)

    #     return

    def get_stats(self, y_test, y_pred):
        """Calculate stats for the current model."""
        stats = {
            'RMSE': f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}",
            'Variance': f"{r2_score(y_test, y_pred):.3f}"
        }
        return stats

    def prediction(self, input_data):
        
        y_pred = self.model.predict(input_data)

        return

    def simpleRNN_(self):
        """Define and return a SimpleRNN model."""
        num_units = self.params.get('num_units', {}).get('value', 10)
        l1_reg = self.params.get('l1_reg', {}).get('value', 0.0)
        seed = self.params.get('seed', {}).get('value', 0)
        activation = self.params.get('activation', {}).get('value', 'tanh')

        model = Sequential()
        model.add(SimpleRNN(
            num_units,
            activation=activation,
            kernel_initializer=keras.initializers.glorot_uniform(seed),
            bias_initializer=keras.initializers.glorot_uniform(seed),
            recurrent_initializer=keras.initializers.orthogonal(seed),
            kernel_regularizer=l1(l1_reg),
            input_shape=(self.data['x_train'].shape[1], 1),
            unroll=True,
            stateful=False
        ))
        model.add(Dense(
            1,
            kernel_initializer=keras.initializers.glorot_uniform(seed),
            bias_initializer=keras.initializers.glorot_uniform(seed),
            kernel_regularizer=l1(l1_reg)
        ))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def lstm_(self):
        """Define and return an LSTM model."""
        num_units = self.params.get('num_units', {}).get('value', 10)
        l1_reg = self.params.get('l1_reg', {}).get('value', 0.0)
        seed = self.params.get('seed', {}).get('value', 0)
        activation = self.params.get('activation', {}).get('value', 'tanh')
        loss = self.params.get('loss', {}).get('value', 'mean_squared_error')
        optimizer = self.params.get('optimizer', {}).get('value', 'adam')
        
        print("num_units:", type(num_units), num_units)
        print("l1_reg:", type(l1_reg), l1_reg)
        print("seed:", type(seed), seed)
        print("activation:", type(activation), activation)
        
        print("In model, parameters loaded")
        try:
            model = Sequential()
            print("First add")
            model.add(LSTM(
                num_units,
                activation=activation,
                kernel_initializer=keras.initializers.glorot_uniform(seed),
                bias_initializer=keras.initializers.glorot_uniform(seed),
                recurrent_initializer=keras.initializers.orthogonal(seed),
                kernel_regularizer=l1(l1_reg),
                input_shape=(self.data['x_train'].shape[1], 1),
                unroll=True
            ))
            print("Second add")
            model.add(Dense(
                1,
                kernel_initializer=keras.initializers.glorot_uniform(seed),
                bias_initializer=keras.initializers.glorot_uniform(seed),
                kernel_regularizer=l1(l1_reg)
            ))
        except Exception as e:
            print(f"Error during model.fit: {str(e)}")
            raise  # Re-raise the exception after logging it
        print("In model, compiling model")
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def gru_(self):
        """Define and return a GRU model."""
        num_units = self.params.get('num_units', {}).get('value', 10)
        l1_reg = self.params.get('l1_reg', {}).get('value', 0.0)
        seed = self.params.get('seed', {}).get('value', 0)
        activation = self.params.get('activation', {}).get('value', 'tanh')

        model = Sequential()
        model.add(GRU(
            num_units,
            activation=activation,
            kernel_initializer=keras.initializers.glorot_uniform(seed),
            bias_initializer=keras.initializers.glorot_uniform(seed),
            recurrent_initializer=keras.initializers.orthogonal(seed),
            kernel_regularizer=l1(l1_reg),
            input_shape=(self.data['x_train'].shape[1], 1),
            unroll=True
        ))
        model.add(Dense(
            1,
            kernel_initializer=keras.initializers.glorot_uniform(seed),
            bias_initializer=keras.initializers.glorot_uniform(seed),
            kernel_regularizer=l1(l1_reg)
        ))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def alpharnn_(self):
        """Define and return an AlphaRNN model."""
        num_units = self.params.get('num_units', {}).get('value', 10)
        l1_reg = self.params.get('l1_reg', {}).get('value', 0.0)
        seed = self.params.get('seed', {}).get('value', 0)
        activation = self.params.get('activation', {}).get('value', 'tanh')

        model = Sequential()
        model.add(AlphaRNN(
            num_units,
            activation=activation,
            kernel_initializer=keras.initializers.glorot_uniform(seed),
            bias_initializer=keras.initializers.glorot_uniform(seed),
            recurrent_initializer=keras.initializers.orthogonal(seed),
            kernel_regularizer=l1(l1_reg),
            input_shape=(self.data['x_train'].shape[1], 1),
            unroll=True
        ))
        model.add(Dense(
            1,
            kernel_initializer=keras.initializers.glorot_uniform(seed),
            bias_initializer=keras.initializers.glorot_uniform(seed),
            kernel_regularizer=l1(l1_reg)
        ))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def alphatrnn_(self):
        """Define and return an AlphatRNN model."""
        num_units = self.params.get('num_units', {}).get('value', 10)
        l1_reg = self.params.get('l1_reg', {}).get('value', 0.0)
        seed = self.params.get('seed', {}).get('value', 0)
        activation = self.params.get('activation', {}).get('value', 'tanh')

        model = Sequential()
        model.add(AlphatRNN(
            num_units,
            activation=activation,
            recurrent_activation='sigmoid',
            kernel_initializer=keras.initializers.glorot_uniform(seed),
            bias_initializer=keras.initializers.glorot_uniform(seed),
            recurrent_initializer=keras.initializers.orthogonal(seed),
            kernel_regularizer=l1(l1_reg),
            input_shape=(self.data['x_train'].shape[1], 1),
            unroll=True
        ))
        model.add(Dense(
            1,
            kernel_initializer=keras.initializers.glorot_uniform(seed),
            bias_initializer=keras.initializers.glorot_uniform(seed),
            kernel_regularizer=l1(l1_reg)
        ))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def ML(self, model_function):
        """Train the model using the provided model function."""
        epochs = self.params.get('epochs', {}).get('value', 201)
        batch_size = self.params.get('batch_size', {}).get('value', 1000)
        print("In model, parameters loaded")

        x_train = self.data['x_train'].values.reshape(self.data['x_train'].shape[0], self.data['x_train'].shape[1], 1)
        es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        print("In model, data transformed")

        self.model = model_function()
        print("In model, model_function specified")

        try:
            self.model.fit(x_train, self.data['y_train'], epochs=epochs, batch_size=batch_size, callbacks=[es], shuffle=False)
        except Exception as e:
            print(f"Error during model.add: {str(e)}")
            raise  # Re-raise the exception after logging it
        print("In model, model fit")

        return

    # def lob_cnn_ffnn(self):
    #     """Define and return a CNN + FFNN model for LOB data."""

    #     # Define input for price and volume
    #     price_input = Input(shape=(5, 1), name="price_input")  # 5 levels of prices
    #     volume_input = Input(shape=(5, 1), name="volume_input")  # 5 levels of volumes

    #     # Convolution for price
    #     price_conv = Conv1D(filters=16, kernel_size=2, activation='relu', name="price_conv")(price_input)
    #     price_flatten = Flatten(name="price_flatten")(price_conv)

    #     # Convolution for volume
    #     volume_conv = Conv1D(filters=16, kernel_size=2, activation='relu', name="volume_conv")(volume_input)
    #     volume_flatten = Flatten(name="volume_flatten")(volume_conv)

    #     # Concatenate price and volume features
    #     combined = Concatenate(name="concat")([price_flatten, volume_flatten])

    #     # Feed-forward layers
    #     dense_1 = Dense(64, activation='relu', name="dense_1")(combined)
    #     dense_2 = Dense(32, activation='relu', name="dense_2")(dense_1)
    #     output = Dense(1, activation='linear', name="output")(dense_2)

    #     # Create the model
    #     model = Model(inputs=[price_input, volume_input], outputs=output, name="LOB_CNN_FFNN")
    #     model.compile(optimizer='adam', loss='mean_squared_error')

    #     return model

    def lob_cnn_ffnn_improved(self):
        """
        Define and return an improved CNN + FFNN model for LOB data.
        This model properly handles price and volume at each level as features.
        """
        print("Entering lob cnn ffnn improved")

        # For each level (bid and ask), we have both price and volume
        # So for each side (bid/ask), each level has 2 features
        
        # Define inputs for bid side (5 levels, each with price and volume)
        bid_input = Input(shape=(5, 2), name="bid_input")  # Shape: [levels, features(price,volume)]
        
        # Define inputs for ask side (5 levels, each with price and volume)
        ask_input = Input(shape=(5, 2), name="ask_input")  # Shape: [levels, features(price,volume)]
        
        # Convolution for bid side - convolving across levels with price and volume as features
        bid_conv = Conv1D(filters=8, kernel_size=1, activation='relu', name="bid_conv")(bid_input)
        bid_flatten = Flatten(name="bid_flatten")(bid_conv)
        
        # Convolution for ask side - convolving across levels with price and volume as features
        ask_conv = Conv1D(filters=8, kernel_size=1, activation='relu', name="ask_conv")(ask_input)
        ask_flatten = Flatten(name="ask_flatten")(ask_conv)
        
        # Concatenate bid and ask features
        combined = Concatenate(name="concat")([bid_flatten, ask_flatten])
        
        # Feed-forward layers
        dense_1 = Dense(64, activation='relu', name="dense_1")(combined)
        dense_2 = Dense(32, activation='relu', name="dense_2")(dense_1)
        output = Dense(1, activation='linear', name="output")(dense_2)
        
        # Create the model
        model = Model(inputs=[bid_input, ask_input], outputs=output, name="LOB_CNN_FFNN_Improved")
        model.compile(optimizer='adam', loss='mean_squared_error')

        print("Exiting lob cnn ffnn improved")
        return model

    def lob_cnn(self):
        """
        Define and return an CNN + FFNN model for LOB data.
        This model properly handles price and volume at each level as features.
        """
        print("Entering lob cnn")

        # For each level (bid and ask), we have both price and volume
        # So for each side (bid/ask), each level has 2 features
        
        # Define inputs for bid side (5 levels, each with price and volume)
        input = Input(shape=(10, 2), name="input")  # Shape: [levels, features(price,volume)]
        
        # Define inputs for ask side (5 levels, each with price and volume)
        # ask_input = Input(shape=(5, 2), name="ask_input")  # Shape: [levels, features(price,volume)]
        
        # Convolution for bid side - convolving across levels with price and volume as features
        conv = Conv1D(filters=8, kernel_size=1, activation='relu', name="conv")(input)
        flatten = Flatten(name="flatten")(conv)
        
        # Convolution for ask side - convolving across levels with price and volume as features
        # ask_conv = Conv1D(filters=8, kernel_size=1, activation='relu', name="ask_conv")(ask_input)
        # ask_flatten = Flatten(name="ask_flatten")(ask_conv)
        
        # Concatenate bid and ask features
        # combined = Concatenate(name="concat")([bid_flatten, ask_flatten])
        
        # Feed-forward layers
        dense_1 = Dense(64, activation='relu', name="dense_1")(flatten)
        dense_2 = Dense(32, activation='relu', name="dense_2")(dense_1)
        output = Dense(1, activation='linear', name="output")(dense_2)
        
        # Create the model
        model = Model(inputs=input, outputs=output, name="LOB_CNN")
        model.compile(optimizer='adam', loss='mean_squared_error')

        print("Exiting lob cnn")
        return model


    def prepare_lob_data_for_cnn(self):
        """
        Prepare LOB data for the improved CNN model.
        This function reshapes the data so price and volume are treated as features at each level.
        
        Returns:
            tuple: (bid_train, ask_train, target_train, bid_test, ask_test, target_test)
        """
        print("Entering prepare lob data for cnn")
        if self.data is None:
            raise ValueError("No LOB data loaded.")
        
        # Extract bid price and volume columns
        bid_price_cols = [f'bid_price_{i}' for i in range(5)]
        bid_volume_cols = [f'bid_volume_{i}' for i in range(5)]

        # Extract ask price and volume columns
        ask_price_cols = [f'ask_price_{i}' for i in range(5)]
        ask_volume_cols = [f'ask_volume_{i}' for i in range(5)]

        # print(bid_price_cols)

        # print(self.data['x_train'])

        # Prepare train data
        bid_prices_train = self.data['x_train'][bid_price_cols].values
        bid_volumes_train = self.data['x_train'][bid_volume_cols].values
        ask_prices_train = self.data['x_train'][ask_price_cols].values
        ask_volumes_train = self.data['x_train'][ask_volume_cols].values
        target_train = self.data['y_train'].values

        print(bid_prices_train)

        # Prepare test data
        bid_prices_test = self.data['x_test'][bid_price_cols].values
        bid_volumes_test = self.data['x_test'][bid_volume_cols].values
        ask_prices_test = self.data['x_test'][ask_price_cols].values
        ask_volumes_test = self.data['x_test'][ask_volume_cols].values
        target_test = self.data['y_test'].values
        
        # Number of samples
        n_train_samples = len(self.data['x_train'])
        n_test_samples = len(self.data['x_test'])
        
        print("n_train_samples", n_train_samples)

        # Reshape train data
        bid_train = np.zeros((n_train_samples, 5, 2))  # [samples, levels, features(price,volume)]
        ask_train = np.zeros((n_train_samples, 5, 2))  # [samples, levels, features(price,volume)]
        for i in range(5):  # For each level
            bid_train[:, i, 0] = bid_prices_train[:, i]  # Price at level i
            bid_train[:, i, 1] = bid_volumes_train[:, i]  # Volume at level i
            ask_train[:, i, 0] = ask_prices_train[:, i]  # Price at level i
            ask_train[:, i, 1] = ask_volumes_train[:, i]  # Volume at level i

        print(bid_train)

        # Reshape test data
        bid_test = np.zeros((n_test_samples, 5, 2))  # [samples, levels, features(price,volume)]
        ask_test = np.zeros((n_test_samples, 5, 2))  # [samples, levels, features(price,volume)]
        for i in range(5):  # For each level
            bid_test[:, i, 0] = bid_prices_test[:, i]  # Price at level i
            bid_test[:, i, 1] = bid_volumes_test[:, i]  # Volume at level i
            ask_test[:, i, 0] = ask_prices_test[:, i]  # Price at level i
            ask_test[:, i, 1] = ask_volumes_test[:, i]  # Volume at level i
        print("Exiting prepare lob data for cnn")        
        return bid_train, ask_train, target_train, bid_test, ask_test, target_test

    def prepare_cnn_input(self, input_series):
        """
        Prepare LOB data for the improved CNN model.
        This function reshapes the data so price and volume are treated as features at each level.
        
        Returns:
            tuple: (x_train, x_test)
        """
        print("Entering prepare cnn input")
        if input_series is None:
            raise ValueError("No input data.")
        
        # Extract bid price and volume columns
        bid_price_cols = [f'bid_price_{i}' for i in range(5)]
        bid_volume_cols = [f'bid_volume_{i}' for i in range(5)]

        # Extract ask price and volume columns
        ask_price_cols = [f'ask_price_{i}' for i in range(5)]
        ask_volume_cols = [f'ask_volume_{i}' for i in range(5)]

        # print(bid_price_cols)

        # Prepare data
        bid_prices = input_series[bid_price_cols].values
        bid_volumes = input_series[bid_volume_cols].values
        ask_prices = input_series[ask_price_cols].values
        ask_volumes = input_series[ask_volume_cols].values

        # print(bid_prices)

        # Number of samples
        n_samples = len(input_series['bid_price_0'])
        
        # print("n_samples", n_samples)

        # Reshape train data
        bid_series = np.zeros((n_samples, 5, 2))  # [samples, levels, features(price,volume)]
        ask_series = np.zeros((n_samples, 5, 2))  # [samples, levels, features(price,volume)]
        for i in range(5):  # For each level
            bid_series[:, i, 0] = bid_prices[:, i]  # Price at level i
            bid_series[:, i, 1] = bid_volumes[:, i]  # Volume at level i
            ask_series[:, i, 0] = ask_prices[:, i]  # Price at level i
            ask_series[:, i, 1] = ask_volumes[:, i]  # Volume at level i

        # print(bid_series)

        x_series = np.concatenate((bid_series, ask_series), axis=1)

        print("Exiting prepare cnn input")        

        return x_series


    def train_lob_cnn(self):
        """
        Train the improved LOB CNN model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Proportion of data to use for validation
            
        Returns:
            history: Training history
        """
        print("Entering train lob cnn")
        epochs = self.params.get('epochs', {}).get('value', 50)
        batch_size = self.params.get('batch_size', {}).get('value', 32)
        validation_split = self.params.get('validation_split', {}).get('value', 0.2)

        # Prepare data
        cnn_x_train = self.prepare_cnn_input(self.data['x_train'])
        cnn_x_test = self.prepare_cnn_input(self.data['x_test'])
        
        # print("cnn_x_train")
        # print(cnn_x_train)

        # print("cnn_x_test")
        # print(cnn_x_test)

        print("Exiting train lob cnn")

        # return

        # bid_train, ask_train, target_train, bid_test, ask_test, target_test = self.prepare_lob_data_for_cnn()

        # bid_train, ask_train, target_train, bid_test, ask_test, target_test = self.prepare_lob_data_for_cnn()
        print("In train_lob_cnn")
        # print(bid_train)
        # print(ask_train)
        # print(target_train)
        # print(bid_test)
        # print(ask_test)
        # print(target_test)

        # Create model
        self.model = self.lob_cnn()
        # self.model = self.lob_cnn_ffnn_improved()
        
        # Print model summary
        self.model.summary()
        
        # print("x shape: ", cnn_x_train.shape)
        # print("y shape: ", self.data["y_train"].shape)

        # Train the model
        history = self.model.fit(
            cnn_x_train,
            self.data["y_train"],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # history = self.model.fit(
        #     [bid_train, ask_train],
        #     target_train,
        #     epochs=epochs,
        #     batch_size=batch_size,
        #     validation_split=validation_split,
        #     verbose=1
        # )

        # Evaluate on test data
        # test_loss = self.model.evaluate([bid_test, ask_test], target_test, verbose=1)
        # print(f"Test Loss: {test_loss}")
        
        print("Exiting train lob cnn")
        # self.model = model
        return