import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.callbacks import EarlyStopping

class ModelHandler:
    def __init__(self, df, train_date='2022-12-31'):
        self.df = df
        self.train_date = train_date
        self.train_data = None
        self.test_data = None
        self.split_data()
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.features = []
        self.target = [] 
        self.model = None

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
        for i in range(1, num_lags + 1):
            self.df['min_' + str(i) + '_close'] = self.df['close'].shift(i)
        # self.df['min_1_close'] = self.df['close'].shift(1)
        # self.df['min_2_close'] = self.df['close'].shift(2)
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

    def regression(self):
        """Perform simple regression."""
        self.create_lagged_features(2)
        print(self.df.columns)

        self.features = ['min_1_close', 'min_2_close']
        self.target = ['close']
        self.prepare_features()

        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(self.X_train, self.Y_train)
        
        return

    def regression_on_trend(self):
        """Perform regression on trend."""
        self.create_trend_features(3)
 
        # self.df['min_1_close'] = self.df['close'].shift(1)
        # self.df['min_2_close'] = self.df['close'].shift(2)
        # self.df['min_3_close'] = self.df['close'].shift(3)
        # self.df['min_4_close'] = self.df['close'].shift(4)
        # self.df['trend_3_day'] = np.where(
        #     (self.df['min_1_close'] > self.df['min_2_close']) +
        #     (self.df['min_2_close'] > self.df['min_3_close']) +
        #     (self.df['min_3_close'] > self.df['min_4_close']) > 0, 1, -1
        # )
        print(self.df.columns)
        self.features = ['min_1_close', 'trend_3_day']
        self.target = ['close']

        # # X_train, X_test, y_train, y_test = self.prepare_features(features, target)
        # self.prepare_features(features, target)
        # model = LinearRegression(fit_intercept=True)
        # model.fit(X_train, y_train)

        # self.features = ['min_1_close', 'min_2_close']
        # self.target = ['close']
        self.prepare_features()

        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(self.X_train, self.Y_train)


        # y_pred = model.predict(X_test)

        # stats = {
        #     'RMSE': f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}",
        #     'Variance': f"{r2_score(y_test, y_pred):.3f}"
        # }

        # plt.scatter(y_test, y_pred)
        # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
        # plt.xlabel('Actual')
        # plt.ylabel('Predicted')
        # plt.legend()

        # return stats, plt

        return

    def ml_regression(self, model_type="rnn", do_training=False):
        """Perform machine learning regression."""
        # Example implementation for RNN
        self.features = ['close']
        self.target = ['close']
        self.prepare_features()
        # X_train, X_test, y_train, y_test = self.prepare_features(features, target)

        self.model = Sequential()
        self.model.add(SimpleRNN(10, activation='tanh', input_shape=(self.X_train.shape[1], 1)))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        if do_training:
            es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            self.model.fit(self.X_train, self.Y_train, epochs=50, batch_size=32, callbacks=[es])

        # y_pred = model.predict(X_test)
        # stats = {
        #     'RMSE': f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}",
        #     'Variance': f"{r2_score(y_test, y_pred):.3f}"
        # }

        # plt.plot(y_test, label='Actual')
        # plt.plot(y_pred, label='Predicted')
        # plt.legend()

        # return stats, plt

        return

    def get_stats(self):
        """Calculate stats for the current model."""
        y_pred = self.model.predict(self.X_test)

        stats = {
            'RMSE': f"{np.sqrt(mean_squared_error(self.Y_test, y_pred)):.3f}",
            'Variance': f"{r2_score(self.Y_test, y_pred):.3f}"
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





