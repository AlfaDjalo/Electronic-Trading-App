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

class ModelHandler:
    def __init__(self, data, params):
    # def __init__(self, df, train_date='2022-12-31'):
        self.data = data
        self.params = params
        # self.df = df
        # self.train_date = train_date
        # self.train_data = None
        # self.test_data = None
        # self.split_data()
        # self.X_train = None
        # self.Y_train = None
        # self.X_test = None
        # self.Y_test = None
        # self.features = []
        # self.target = [] 
        # self.model = None

    def split_data(self):
        """Split the data into training and testing sets based on the train_date."""
        self.df.dropna(inplace=True)
        self.df = self.df.reset_index()  # Moves index columns back to normal columns
        # print(self.df.columns)
        
        train_mask = self.df['date'] <= self.train_date
        test_mask = self.df['date'] > self.train_date
        self.train_data = self.df[train_mask]
        self.test_data = self.df[test_mask]

    def create_lagged_features(self, num_lags):
        """Create lagged features for the regression model."""
        if num_lags > 0:        
            for i in range(1, num_lags + 1):
                self.df['min_' + str(i) + '_close'] = self.df['close'].shift(i)
        elif num_lags < 0:
            self.df[f'fut_{-num_lags}_close'] = self.df['close'].shift(num_lags)

        self.df.dropna(inplace=True)  # Drop rows with NaN values created by shifting

        # Update train_data and test_data with the new features
        train_mask = self.df['date'] <= self.train_date
        test_mask = self.df['date'] > self.train_date
        self.train_data = self.df[train_mask]
        self.test_data = self.df[test_mask]

    def create_trend_features(self, num_lags):
        """Create lagged features for the regression model."""
        self.create_lagged_features(num_lags+1)

        overall_trend = 0
        for i in range(1, num_lags + 1):
            self.df['min_' + str(i) + '_trend'] = np.where(self.df['min_' + str(i) + '_close'] > self.df['min_' + str(i+1) + '_close'], 1, -1)
            overall_trend += self.df['min_' + str(i) + '_trend']

        self.df['trend_' + str(num_lags) + '_day'] = np.where(overall_trend > 0, 1, -1)

        # self.df['min_1_close'] = self.df['close'].shift(1)
        # self.df['min_2_close'] = self.df['close'].shift(2)
        self.df.dropna(inplace=True)  # Drop rows with NaN values created by shifting

        # Update train_data and test_data with the new features
        train_mask = self.df['date'] <= self.train_date
        test_mask = self.df['date'] > self.train_date
        self.train_data = self.df[train_mask]
        self.test_data = self.df[test_mask]

    def prepare_features(self):
    # def prepare_features(self, features, target):
        """Prepare features and target for training and testing."""
        self.df.dropna(inplace=True)
        self.X_train = self.train_data[self.features]
        self.X_test = self.test_data[self.features]
        self.Y_train = self.train_data[self.target]
        self.Y_test = self.test_data[self.target]
        return # X_train, X_test, y_train, y_test

    def standardise_input(self, feature, drop=False):

        # mu = float(self.X_train[feature].mean())
        # sigma = float(self.X_train[feature].std())
        mu = float(self.X_train[feature].iloc[0])
        sigma = float(self.X_train[feature].iloc[0])

        stdize_input = lambda x: (x - mu) / sigma

        # X_train = X_train.apply(stdize_input)
        # X_test = X_test.apply(stdize_input)

        self.X_train = (self.X_train - mu) / sigma
        self.X_test = (self.X_test - mu) / sigma

        if drop==True:
            self.X_train[feature].drop
            self.X_test[feature].drop
        
        return

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

    def get_stats(self):
        """Calculate stats for the current model."""
        y_pred = self.model.predict(self.data['x_test'])
        # y_pred = self.model.predict(self.X_test)

        stats = {
            'RMSE': f"{np.sqrt(mean_squared_error(self.data['y_test'], y_pred)):.3f}",
            'Variance': f"{r2_score(self.data['y_test'], y_pred):.3f}"
            # 'RMSE': f"{np.sqrt(mean_squared_error(self.Y_test, y_pred)):.3f}",
            # 'Variance': f"{r2_score(self.Y_test, y_pred):.3f}"
        }
        return stats

    def get_plt(self):
        """Create a plot for the current model."""
        y_pred = self.model.predict(self.X_test) 

        plt.scatter(self.Y_test, y_pred)
        plt.plot([self.Y_test.min(), self.Y_test.max()], [self.Y_test.min(), self.Y_test.max()], 'r--', label='Perfect Fit')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()

        return plt

    def prediction(self, input_data):
        
        y_pred = self.model.predict(input_data)

        return
