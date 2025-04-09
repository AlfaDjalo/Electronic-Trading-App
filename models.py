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

from alphaRNN import AlphaRNN
from alphatRNN import AlphatRNN  # Import AlphaRNN and AlphatRNN modules

class ModelHandler:
    def __init__(self, data, params):
        self.data = data
        self.params = params


    # def split_data(self):
    #     """Split the data into training and testing sets based on the train_date."""
    #     self.df.dropna(inplace=True)
    #     self.df = self.df.reset_index()  # Moves index columns back to normal columns
    #     # print(self.df.columns)
        
    #     train_mask = self.df['date'] <= self.train_date
    #     test_mask = self.df['date'] > self.train_date
    #     self.train_data = self.df[train_mask]
    #     self.test_data = self.df[test_mask]

    # def create_lagged_features(self, num_lags):
    #     """Create lagged features for the regression model."""
    #     if num_lags > 0:        
    #         for i in range(1, num_lags + 1):
    #             self.df['min_' + str(i) + '_close'] = self.df['close'].shift(i)
    #     elif num_lags < 0:
    #         self.df[f'fut_{-num_lags}_close'] = self.df['close'].shift(num_lags)

    #     self.df.dropna(inplace=True)  # Drop rows with NaN values created by shifting

    #     # Update train_data and test_data with the new features
    #     train_mask = self.df['date'] <= self.train_date
    #     test_mask = self.df['date'] > self.train_date
    #     self.train_data = self.df[train_mask]
    #     self.test_data = self.df[test_mask]

    # def create_trend_features(self, num_lags):
    #     """Create lagged features for the regression model."""
    #     self.create_lagged_features(num_lags+1)

    #     overall_trend = 0
    #     for i in range(1, num_lags + 1):
    #         self.df['min_' + str(i) + '_trend'] = np.where(self.df['min_' + str(i) + '_close'] > self.df['min_' + str(i+1) + '_close'], 1, -1)
    #         overall_trend += self.df['min_' + str(i) + '_trend']

    #     self.df['trend_' + str(num_lags) + '_day'] = np.where(overall_trend > 0, 1, -1)

    #     # self.df['min_1_close'] = self.df['close'].shift(1)
    #     # self.df['min_2_close'] = self.df['close'].shift(2)
    #     self.df.dropna(inplace=True)  # Drop rows with NaN values created by shifting

    #     # Update train_data and test_data with the new features
    #     train_mask = self.df['date'] <= self.train_date
    #     test_mask = self.df['date'] > self.train_date
    #     self.train_data = self.df[train_mask]
    #     self.test_data = self.df[test_mask]

    # def prepare_features(self):
    # # def prepare_features(self, features, target):
    #     """Prepare features and target for training and testing."""
    #     self.df.dropna(inplace=True)
    #     self.X_train = self.train_data[self.features]
    #     self.X_test = self.test_data[self.features]
    #     self.Y_train = self.train_data[self.target]
    #     self.Y_test = self.test_data[self.target]
    #     return # X_train, X_test, y_train, y_test

    # def standardise_input(self, feature, drop=False):

    #     # mu = float(self.X_train[feature].mean())
    #     # sigma = float(self.X_train[feature].std())
    #     mu = float(self.X_train[feature].iloc[0])
    #     sigma = float(self.X_train[feature].iloc[0])

    #     stdize_input = lambda x: (x - mu) / sigma

    #     # X_train = X_train.apply(stdize_input)
    #     # X_test = X_test.apply(stdize_input)

    #     self.X_train = (self.X_train - mu) / sigma
    #     self.X_test = (self.X_test - mu) / sigma

    #     if drop==True:
    #         self.X_train[feature].drop
    #         self.X_test[feature].drop
        
    #     return

    def regression(self):
        """Perform simple regression."""
        # self.create_lagged_features(2)
        # print(self.df.columns)

        # self.features = ['min_1_close', 'min_2_close']
        # self.target = ['close']
        # self.prepare_features()

        # print('x_train')
        # print(self.data['x_train'])

        # print('y_train')
        # print(self.data['y_train'])

        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(self.data['x_train'], self.data['y_train'])

        return

    # def regression(self):
    #     """Perform simple regression."""
    #     self.create_lagged_features(2)
    #     print(self.df.columns)

    #     self.features = ['min_1_close', 'min_2_close']
    #     self.target = ['close']
    #     self.prepare_features()

    #     self.model = LinearRegression(fit_intercept=True)
    #     self.model.fit(self.X_train, self.Y_train)
        
    #     return

    def regression_on_trend(self):
        """Perform regression on trend."""
        self.create_trend_features(3)
 

        print(self.df.columns)
        self.features = ['min_1_close', 'trend_3_day']
        self.target = ['close']

        self.prepare_features()

        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(self.X_train, self.Y_train)

        return

    def rnn(self):
        """Perform rnn regression."""
        # if parameters is None:
        #     parameters = {}

        # print("In rnn model")

        # print(self.params)
        # Extract parameters
        epochs = self.params.get('epochs', {}).get('value', 201)
        batch_size = self.params.get('batch_size', {}).get('value', 1000)
        num_units = self.params.get('num_units', {}).get('value', 10)
        l1_reg = self.params.get('l1_reg', {}).get('value', 0.0)
        seed = self.params.get('seed', {}).get('value', 0)
        activation = self.params.get('activation', {}).get('value', 'tanh')
        # print("Parameters extracted")

        # self.create_lagged_features(4)    
        # self.create_lagged_features(-4)
        # print(self.df.columns)

        # self.features = ['close', 'min_1_close', 'min_2_close', 'min_3_close', 'min_4_close']
        # self.target = ['fut_4_close']
        # self.prepare_features()

        # self.standardise_input(['close'], drop=True)

        x_train = self.data['x_train'].values.reshape(self.data['x_train'].shape[0], self.data['x_train'].shape[1], 1)
        # print(x_train)

        def SimpleRNN_():
            model = Sequential()
            model.add(SimpleRNN(num_units, activation=activation, kernel_initializer=keras.initializers.glorot_uniform(seed), \
                bias_initializer=keras.initializers.glorot_uniform(seed), recurrent_initializer=keras.initializers.orthogonal(seed), \
                kernel_regularizer=l1(l1_reg), input_shape=(x_train.shape[1], x_train.shape[-1]), unroll=True, stateful=False))  
            model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), \
                kernel_regularizer=l1(l1_reg)))
            model.compile(loss='mean_squared_error', optimizer='adam')
            return model

        # def LSTM_():
        #     model = Sequential()
        #     model.add(LSTM(num_units, activation='tanh', kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), recurrent_initializer=keras.initializers.orthogonal(seed), kernel_regularizer=l1(l1_reg), input_shape=(self.X_train.shape[1], self.X_train.shape[-1]), unroll=True)) 
        #     model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), kernel_regularizer=l1(l1_reg)))
        #     model.compile(loss='mean_squared_error', optimizer='adam')
        #     return model

        # params = {
        #     'rnn': {'function': SimpleRNN_},
        #     'lstm': {'function': LSTM_}
        # }

        # model_chosen = model_type
        es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        # print("About to run model")

        tf.random.set_seed(seed)
        # print('Training rnn model')
        self.model = SimpleRNN_()
        self.model.fit(x_train, self.data['y_train'], epochs=epochs, 
                  batch_size=batch_size, callbacks=[es], shuffle=False)
        # print("Just run model")

        return

    # def ML(self, model_name):
    #     """Perform rnn regression."""
    #     epochs = self.params.get('epochs', {}).get('value', 201)
    #     batch_size = self.params.get('batch_size', {}).get('value', 1000)
    #     num_units = self.params.get('num_units', {}).get('value', 10)
    #     l1_reg = self.params.get('l1_reg', {}).get('value', 0.0)
    #     seed = self.params.get('seed', {}).get('value', 0)
    #     activation = self.params.get('activation', {}).get('value', 'tanh')

    #     x_train = self.data['x_train'].values.reshape(self.data['x_train'].shape[0], self.data['x_train'].shape[1], 1)

    #     es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    #     tf.random.set_seed(seed)
    #     self.model = model_name()
    #     self.model.fit(x_train, self.data['y_train'], epochs=epochs, 
    #               batch_size=batch_size, callbacks=[es], shuffle=False)

    #     return

    def ml_regression(self, model_type="rnn", do_training=False, parameters=None):
        """Perform machine learning regression."""
        if parameters is None:
            parameters = {}

        # Extract parameters
        epochs = parameters.get('epochs', 2000)
        batch_size = parameters.get('batch_size', 1000)
        n_units = parameters.get('n_units', 10)
        l1_reg = parameters.get('l1_reg', 0.0)
        seed = parameters.get('seed', 0)

        self.create_lagged_features(4)    
        self.create_lagged_features(-4)
        print(self.df.columns)

        self.features = ['close', 'min_1_close', 'min_2_close', 'min_3_close', 'min_4_close']
        self.target = ['fut_4_close']
        self.prepare_features()

        self.standardise_input(['close'], drop=True)

        self.X_train = self.X_train.values.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)

        def SimpleRNN_():
            model = Sequential()
            model.add(SimpleRNN(n_units, activation='tanh', kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), recurrent_initializer=keras.initializers.orthogonal(seed), kernel_regularizer=l1(l1_reg), input_shape=(self.X_train.shape[1], self.X_train.shape[-1]), unroll=True, stateful=False))  
            model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), kernel_regularizer=l1(l1_reg)))
            model.compile(loss='mean_squared_error', optimizer='adam')
            return model

        def LSTM_():
            model = Sequential()
            model.add(LSTM(n_units, activation='tanh', kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), recurrent_initializer=keras.initializers.orthogonal(seed), kernel_regularizer=l1(l1_reg), input_shape=(self.X_train.shape[1], self.X_train.shape[-1]), unroll=True)) 
            model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), kernel_regularizer=l1(l1_reg)))
            model.compile(loss='mean_squared_error', optimizer='adam')
            return model

        params = {
            'rnn': {'function': SimpleRNN_},
            'lstm': {'function': LSTM_}
        }

        model_chosen = model_type
        es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        tf.random.set_seed(seed)
        print('Training', model_chosen, 'model')
        self.model = params[model_chosen]['function']()
        self.model.fit(self.X_train, self.Y_train, epochs=epochs, 
                  batch_size=batch_size, callbacks=[es], shuffle=False)

        return

    def get_stats(self, y_test, y_pred):
        """Calculate stats for the current model."""
        stats = {
            'RMSE': f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}",
            'Variance': f"{r2_score(y_test, y_pred):.3f}"
        }
        return stats

    # def get_plt(self):
    #     """Create a plot for the current model."""
    #     y_pred = self.model.predict(self.X_test) 

    #     plt.scatter(self.Y_test, y_pred)
    #     plt.plot([self.Y_test.min(), self.Y_test.max()], [self.Y_test.min(), self.Y_test.max()], 'r--', label='Perfect Fit')
    #     plt.xlabel('Actual')
    #     plt.ylabel('Predicted')
    #     plt.legend()

    #     return plt

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

    # def AlphatRNN_(self):
    #     model = Sequential()
    #     model.add(AlphatRNN(n_units, activation='tanh', recurrent_activation='sigmoid', kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), recurrent_initializer=keras.initializers.orthogonal(seed), kernel_regularizer=l1(l1_reg), input_shape=(x_train.shape[1], x_train.shape[-1]), unroll=True))  
    #     model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), kernel_regularizer=l1(l1_reg)))
    #     model.compile(loss='mean_squared_error', optimizer='adam')
    #     return model

    # def AlphaRNN_(self):
    #     model = Sequential()
    #     model.add(AlphaRNN(
    #         n_units,
    #         activation='tanh',
    #         kernel_initializer=keras.initializers.glorot_uniform(seed),
    #         bias_initializer=keras.initializers.glorot_uniform(seed),
    #         recurrent_initializer=keras.initializers.orthogonal(seed),
    #         kernel_regularizer=l1(l1_reg),
    #         input_shape=(self.data['x_train'].shape[1], self.data['x_train'].shape[-1]),
    #         unroll=True
    #     ))
    #     model.add(Dense(
    #         1,
    #         kernel_initializer=keras.initializers.glorot_uniform(seed),
    #         bias_initializer=keras.initializers.glorot_uniform(seed),
    #         kernel_regularizer=l1(l1_reg)
    #     ))
    #     model.compile(loss='mean_squared_error', optimizer='adam')
    #     return model

    # def GRU_(self):
    #     model = Sequential()
    #     model.add(GRU(
    #         n_units,
    #         activation='tanh',
    #         kernel_initializer=keras.initializers.glorot_uniform(seed),
    #         bias_initializer=keras.initializers.glorot_uniform(seed),
    #         recurrent_initializer=keras.initializers.orthogonal(seed),
    #         kernel_regularizer=l1(l1_reg),
    #         input_shape=(self.data['x_train'].shape[1], self.data['x_train'].shape[-1]),
    #         unroll=True
    #     ))
    #     model.add(Dense(
    #         1,
    #         kernel_initializer=keras.initializers.glorot_uniform(seed),
    #         bias_initializer=keras.initializers.glorot_uniform(seed),
    #         kernel_regularizer=l1(l1_reg)
    #     ))
    #     model.compile(loss='mean_squared_error', optimizer='adam')
    #     return model