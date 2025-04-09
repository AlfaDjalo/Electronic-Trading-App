import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

from arch import arch_model

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV

import keras.initializers
from keras.layers import Dense, Layer, LSTM, GRU, SimpleRNN, RNN
from keras.models import Sequential
# from keras.models import load_model
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping

# # from keras.wrappers.scikit_learn import KerasRegressor
# from scikeras.wrappers import KerasRegressor


def regression(df, train_date='2022-12-31'):
    # Compute n-day lag close price within each ticker group
    df['min_1_close'] = df['close'].shift(1)
    df['min_2_close'] = df['close'].shift(2)

    features = ['min_1_close', 'min_2_close']
    target = 'close'

    df.dropna(inplace=True)
    df = df.reset_index()  # Moves index columns back to normal columns
    print(df.columns)

    # Split the data based on the date
    train_mask = df['date'] <= train_date
    test_mask = df['date'] > train_date

    X_train = df.loc[train_mask, features]
    X_test = df.loc[test_mask, features]

    y_train = df.loc[train_mask, target]
    y_test = df.loc[test_mask, target]

    # Check if training set is not empty
    if X_train.empty or X_test.empty:
        print("Error: Training or testing set is empty. Check your date range.")
        return


    # Create linear regression object
    # regr = linear_model.LinearRegression(fit_intercept=False)
    regr = linear_model.LinearRegression(fit_intercept=True)

    # Train the model using the training set
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    y_test_min = np.min(y_test)
    y_test_max = np.max(y_test) 
    y_pred_min = np.min(y_pred)
    y_pred_max = np.max(y_pred) 

    plt.scatter(y_test, y_pred)
    plt.plot([y_test_min, y_test_max], [y_pred_min, y_pred_max], 'r--', label='perfect fit')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()

    stats = {
        'RMSE': f"{np.sqrt(mean_squared_error(y_test, X_test.min_1_close)):.3f}",
        'Variance': f"{r2_score(y_test, y_pred):.3f}"
    }

    return stats, plt


def regression_on_trend(df, train_date='2022-12-31'):
    # Compute n-day lag close price within each ticker group
    df['min_1_close'] = df['close'].shift(1)
    df['min_2_close'] = df['close'].shift(2)
    df['min_3_close'] = df['close'].shift(3)
    df['min_4_close'] = df['close'].shift(4)

    # Compute log returns
    df['log_ret'] = np.log(df['close']) - np.log(df['min_1_close'])

    df['variance'] = df['log_ret'].rolling(180).var()

    df['min_1_trend'] = np.where(df['min_1_close'] > df['min_2_close'], 1, -1)
    df['min_2_trend'] = np.where(df['min_2_close'] > df['min_3_close'], 1, -1)
    df['min_3_trend'] = np.where(df['min_3_close'] > df['min_4_close'], 1, -1)

    df['trend_3_day'] = np.where(df['min_1_trend']+df['min_2_trend']+df['min_3_trend'] > 0, 1, -1)

    features = ['min_1_close', 'trend_3_day']
    target = 'close'

    df.dropna(inplace=True)
    df = df.reset_index()  # Moves index columns back to normal columns

    # Split the data based on the date
    train_mask = df['date'] <= train_date
    test_mask = df['date'] > train_date

    X_train = df.loc[train_mask, features]
    X_test = df.loc[test_mask, features]

    y_train = df.loc[train_mask, target]
    y_test = df.loc[test_mask, target]

    # Check if training set is not empty
    if X_train.empty or X_test.empty:
        print("Error: Training or testing set is empty. Check your date range.")
        return

    # Create linear regression object
    # regr = linear_model.LinearRegression(fit_intercept=False)
    regr = linear_model.LinearRegression(fit_intercept=True)

    # Train the model using the training set
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    y_test_min = np.min(y_test)
    y_test_max = np.max(y_test) 
    y_pred_min = np.min(y_pred)
    y_pred_max = np.max(y_pred) 

    plt.scatter(y_test, y_pred)
    plt.plot([y_test_min, y_test_max], [y_pred_min, y_pred_max], 'r--', label='perfect fit')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()

    stats = {
        'RMSE': f"{np.sqrt(mean_squared_error(y_test, X_test.min_1_close)):.3f}",
        'Variance': f"{r2_score(y_test, y_pred):.3f}"
    }

    return stats, plt


def ML_regression(df, model_chosen, train_date='2022-12-31', do_training=False):
    n_steps_ahead = 4 # forecasting horizon
    n_steps = 4
    print('n_steps set to', n_steps)

    df.dropna(inplace=True)
    df = df.reset_index()  # Moves index columns back to normal columns
    print(df.columns)

    features = ['close']
    target = ['close']

    # Split the data based on the date
    train_mask = df['date'] <= train_date
    test_mask = df['date'] > train_date

    df_train = df.loc[train_mask, features]
    df_test = df.loc[test_mask, features]

    y_train = df.loc[train_mask, target]
    y_test = df.loc[test_mask, target]

    mu = float(df_train.mean())
    sigma = float(df_train.std())

    stdize_input = lambda x: (x - mu) / sigma

    # X_train = X_train.apply(stdize_input)
    # X_test = X_test.apply(stdize_input)

    df_train = (df_train - mu) / sigma
    df_test = (df_test - mu) / sigma

    print(df_train)

    def get_lagged_features(df, n_steps, n_steps_ahead):
        """
        df: pandas DataFrame of time series to be lagged
        n_steps: number of lags, i.e. sequence length
        n_steps_ahead: forecasting horizon
        """
        lag_list = []
        for lag in range(n_steps + n_steps_ahead - 1, n_steps_ahead - 1, -1):
            lag_list.append(df.shift(lag))
        lag_array = np.dstack([i[n_steps+n_steps_ahead-1:] for i in lag_list])
        # We swap the last two dimensions so each slice along the first dimension
        # is the same shape as the corresponding segment of the input time series 
        lag_array = np.swapaxes(lag_array, 1, -1)
        return lag_array

    x_train = get_lagged_features(df_train[features], n_steps, n_steps_ahead)
    y_train =  df_train[target].values[n_steps + n_steps_ahead - 1:]
    y_train_timestamps = df_train.index[n_steps + n_steps_ahead - 1:]

    x_test = get_lagged_features(df_test[features], n_steps, n_steps_ahead)
    y_test =  df_test[target].values[n_steps + n_steps_ahead - 1:]
    y_test_timestamps = df_test.index[n_steps + n_steps_ahead - 1:]

    print([tensor.shape for tensor in (x_train, y_train, x_test, y_test)])

    # Assuming x_train and x_test are your original datasets
    # x_train = np.squeeze(x_train, axis=-1)  # Removes the last dimension
    # x_test = np.squeeze(x_test, axis=-1)    # Removes the last dimension# Assuming x_train and x_test are your original datasets
    # y_train = np.squeeze(y_train, axis=-1)  # Removes the last dimension
    # y_test = np.squeeze(y_test, axis=-1)    # Removes the last dimension

    if x_train.ndim == 2:  # Add a feature dimension if missing
        x_train = np.expand_dims(x_train, axis=-1)
    if x_test.ndim == 2:
        x_test = np.expand_dims(x_test, axis=-1)

    # def SimpleRNN_(n_units = 10, l1_reg=0, seed=0):
    #     model = Sequential()
    #     model.add(SimpleRNN(n_units, activation='tanh', kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), recurrent_initializer=keras.initializers.orthogonal(seed), kernel_regularizer=l1(l1_reg), input_shape=(x_train.shape[1], x_train.shape[-1]), unroll=True, stateful=False))  
    #     model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), kernel_regularizer=l1(l1_reg)))
    #     model.compile(loss='mean_squared_error', optimizer='adam')
    #     return model

    def SimpleRNN_(n_units = 10, l1_reg=0, seed=0):
        print(f"Expected Input Shape: ({x_train.shape[1]}, {x_train.shape[-1]})")  # Debugging
        model = Sequential()
        model.add(SimpleRNN(
            n_units, 
            activation='tanh', 
            kernel_initializer=keras.initializers.glorot_uniform(seed), 
            bias_initializer=keras.initializers.glorot_uniform(seed), 
            recurrent_initializer=keras.initializers.orthogonal(seed), 
            kernel_regularizer=l1(l1_reg), 
            input_shape=(x_train.shape[1], x_train.shape[-1]),  
            unroll=True, 
            stateful=False
        ))  
        model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), kernel_regularizer=l1(l1_reg)))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model


    max_epochs = 200#2000
    batch_size = 1000

    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=50, min_delta=3e-5, restore_best_weights=True)

    params = {
        'rnn': {
            'model': None, 'function': SimpleRNN_, 'l1_reg': 0.0, 'H': 20, 
            'color': 'blue', 'label':'RNN'}, 
        # 'alpharnn': {
        #     'model': None, 'function': AlphaRNN_, 'l1_reg': 0.0, 'H': 10, 
        #     'color': 'green', 'label': '$\\alpha$-RNN' }, 
        # 'alphatrnn': {
        #     'model': None, 'function': AlphatRNN_, 'l1_reg': 0.0, 'H': 5, 
        #     'color': 'cyan', 'label': '$\\alpha_t$-RNN'},
        # 'gru': {
        #     'model': None, 'function': GRU_,'l1_reg': 0.0, 'H': 10, 
        #     'color': 'orange', 'label': 'GRU'},
        # 'lstm': {
        #     'model': None, 'function': LSTM_, 'l1_reg': 0.0, 'H': 10, 
        #     'color':'red', 'label': 'LSTM'}
    }

    # if do_training is True:
    # for key in params.keys():
    tf.random.set_seed(0)
    print('Training', model_chosen, 'model')
    print(f"x_train shape before training: {x_train.shape}")  
    model = params[model_chosen]['function'](params[model_chosen]['H'], params[model_chosen]['l1_reg'])
    model.fit(x_train, y_train, epochs=max_epochs, 
            batch_size=batch_size, callbacks=[es], shuffle=False)
    params[model_chosen]['model'] = model

    # for key in params.keys():
    params[model_chosen]['model'].save('RNNs-SAVED-' + model_chosen + '.keras')  # creates a HDF5 file


# for key in params.keys():
    model = params[model_chosen]['model']
    model.summary()
    
    params[model_chosen]['pred_train'] = model.predict(x_train, verbose=1)
    params[model_chosen]['MSE_train'] = mean_squared_error(y_train, params[model_chosen]['pred_train'])
    
    params[model_chosen]['pred_test'] = model.predict(x_test, verbose=1) 
    params[model_chosen]['MSE_test'] = mean_squared_error(y_test, params[model_chosen]['pred_test'])

    print('training set:', len(y_train))
    print('testing set:', len(y_test))


    max_pts = 10**4
    # compare = params.keys() # e.g. ['rnn', 'alpharnn'] or ['lstm']
    l, u = (None, None) # lower and upper indices of range to plot 
    ds = max(1, len(y_train[l:u])//max_pts) # Downsampling ratio for under `max_pts`
                                        # per series.  Set `None` to disable. 

    fig = plt.figure(figsize=(15,8))
    x_vals = y_train_timestamps[l:u:ds]
    # for key in compare:
    y_vals = params[model_chosen]['pred_train'][l:u:ds]
    label = params[model_chosen]['label'] + ' (train MSE: %.2e)' % params[model_chosen]['MSE_train']
    plt.plot(x_vals, y_vals, c=params[model_chosen]['color'], label=label, lw=1)
    plt.plot(x_vals, y_train[l:u:ds], c="black", label="Observed", lw=1)
    start, end = x_vals.min(), x_vals.max()
    xticks =  [start.date() + timedelta(days=(1+i)) for i in range(1 + (end - start).days)]
    xticks = xticks[::max(1, len(xticks)//30)]
    for t in xticks: plt.axvline(x=t, c='gray', linewidth=0.5, zorder=0)
    plt.xticks(xticks, rotation=70)
    plt.xlim(start, end)
    plt.ylabel('$\hat{Y}$', rotation=0, fontsize=14)
    plt.legend(loc="best", fontsize=12)
    plt.title('Observed vs Model Outputs (Training)', fontsize=16)



    return



def RNN_regression(df, model_chosen, train_date='2022-12-31', do_training=False):
    n_steps_ahead = 4 # forecasting horizon
    n_steps = 4
    print('n_steps set to', n_steps)


    # Split the data based on the date
    train_mask = df['date'] <= train_date
    test_mask = df['date'] > train_date

    X_train = df.loc[train_mask, features]
    X_test = df.loc[test_mask, features]

    y_train = df.loc[train_mask, target]
    y_test = df.loc[test_mask, target]

    X_train = X_train.values.reshape((X_train.shape[0], n_steps, -1))
    X_test = X_test.values.reshape((X_test.shape[0], n_steps, -1))


    mu = float(df_train.mean().iloc[0])
    sigma = float(df_train.std().iloc[0])

    stdize_input = lambda x: (x - mu) / sigma

    df_train = df_train.apply(stdize_input)
    df_test = df_test.apply(stdize_input)



    features = ['min_1_close', 'min_2_close', 'min_3_close', 'min_4_close']
    target = 'close'


    df['min_1_close'] = df['close'].shift(1)
    df['min_2_close'] = df['close'].shift(2)
    df['min_3_close'] = df['close'].shift(3)
    df['min_4_close'] = df['close'].shift(4)



    def SimpleRNN_(n_units = 10, l1_reg=0, seed=0):
        model = Sequential()
        model.add(SimpleRNN(n_units, activation='tanh', kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), recurrent_initializer=keras.initializers.orthogonal(seed), kernel_regularizer=l1(l1_reg), input_shape=(x_train.shape[1], x_train.shape[-1]), unroll=True, stateful=False))  
        model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), kernel_regularizer=l1(l1_reg)))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def AlphatRNN_(n_units = 10, l1_reg=0, seed=0):
        model = Sequential()
        model.add(AlphatRNN(n_units, activation='tanh', recurrent_activation='sigmoid', kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), recurrent_initializer=keras.initializers.orthogonal(seed), kernel_regularizer=l1(l1_reg), input_shape=(x_train.shape[1], x_train.shape[-1]), unroll=True))  
        model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), kernel_regularizer=l1(l1_reg)))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def AlphaRNN_(n_units = 10, l1_reg=0, seed=0):
        model = Sequential()
        model.add(AlphaRNN(n_units, activation='tanh', kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), recurrent_initializer=keras.initializers.orthogonal(seed), kernel_regularizer=l1(l1_reg), input_shape=(x_train.shape[1], x_train.shape[-1]), unroll=True))
        model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), kernel_regularizer=l1(l1_reg)))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def GRU_(n_units = 10, l1_reg=0, seed=0):
        model = Sequential()
        model.add(GRU(n_units, activation='tanh', kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), recurrent_initializer=keras.initializers.orthogonal(seed), kernel_regularizer=l1(l1_reg), input_shape=(x_train.shape[1], x_train.shape[-1]), unroll=True))  
        model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), kernel_regularizer=l1(l1_reg)))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def LSTM_(n_units = 10, l1_reg=0, seed=0):
        model = Sequential()
        model.add(LSTM(n_units, activation='tanh', kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), recurrent_initializer=keras.initializers.orthogonal(seed), kernel_regularizer=l1(l1_reg), input_shape=(x_train.shape[1], x_train.shape[-1]), unroll=True)) 
        model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed), bias_initializer=keras.initializers.glorot_uniform(seed), kernel_regularizer=l1(l1_reg)))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    max_epochs = 1000
    batch_size = 1000

    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=100, min_delta=1e-7, restore_best_weights=True)

    params = {
        'rnn': {
            'model': None, 'function': SimpleRNN_, 'l1_reg': 0.0, 'H': 20, 
            'color': 'blue', 'label':'RNN'}, 
        'alpharnn': {
            'model': None, 'function': AlphaRNN_, 'l1_reg': 0.0, 'H': 10, 
            'color': 'green', 'label': '$\\alpha$-RNN' }, 
        'alphatrnn': {
            'model': None, 'function':AlphatRNN_, 'l1_reg': 0.0, 'H': 5, 
            'color': 'cyan', 'label': '$\\alpha_t$-RNN'},
        'gru': {
            'model': None, 'function':GRU_,'l1_reg': 0.0, 'H': 10, 
            'color': 'orange', 'label': 'GRU'},
        'lstm': {
            'model': None, 'function': LSTM_,'l1_reg': 0.0, 'H': 10, 
            'color':'red', 'label': 'LSTM'}
    }


    # cross_val = False # WARNING: Changing this to True will take many hours to run

    # if do_training and cross_val:
    #     n_units = [5, 10, 20]
    #     l1_reg = [0, 0.001, 0.01, 0.1]
        
    #     # A dictionary containing a list of values to be iterated through
    #     # for each parameter of the model included in the search
    #     param_grid = {'n_units': n_units, 'l1_reg': l1_reg}
        
    #     # In the kth split, TimeSeriesSplit returns first k folds 
    #     # as training set and the (k+1)th fold as test set.
    #     tscv = TimeSeriesSplit(n_splits = 5)
        
    #     # A grid search is performed for each of the models, and the parameter set which
    #     # performs best over all the cross-validation splits is saved in the `params` dictionary

    #     print('Performing cross-validation. Model:', model_chosen)
    #     model = KerasRegressor(build_fn=params[model_chosen]['function'], epochs=max_epochs, 
    #                         batch_size=batch_size, verbose=2)
    #     grid = GridSearchCV(estimator=model, param_grid=param_grid, 
    #                         cv=tscv, n_jobs=1, verbose=2)
    #     grid_result = grid.fit(x_train, y_train, callbacks=[es])
    #     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        
    #     means = grid_result.cv_results_['mean_test_score']
    #     stds = grid_result.cv_results_['std_test_score']
    #     params_ = grid_result.cv_results_['params']
    #     for mean, stdev, param_ in zip(means, stds, params_):
    #         print("%f (%f) with %r" % (mean, stdev, param_))
            
    #     params[model_chosen]['H'] = grid_result.best_params_['n_units']
    #     params[model_chosen]['l1_reg']= grid_result.best_params_['l1_reg']

    # if do_training is True:
    #     tf.random.set_seed(0)
    #     print('Training', model_chosen, 'model')
    #     model = params[model_chosen]['function'](params[model_chosen]['H'], params[model_chosen]['l1_reg'])
    #     model.fit(x_train, y_train, epochs=max_epochs, 
    #             batch_size=batch_size, callbacks=[es], shuffle=False)
    #     params[model_chosen]['model'] = model

    # params[model_chosen]['model'].save('RNNs-Bitcoin-SAVED-' + model_chosen + '.hdf5', overwrite=True)  # creates a HDF5 file

    # def sigmoid(x):
    #     return (1 / (1 + np.exp(-x)))

    # model = params[model_chosen]['model']
    # names = [weight.name for layer in model.layers for weight in layer.weights]
    # weights = model.get_weights()

    # for name, weight in zip(names, weights):
    #     if 'alpha:0' in name:
    #         print("alpha = " + str(sigmoid(weight)))


    # model = params[model_chosen]['model']
    # model.summary()
    
    # params[model_chosen]['pred_train'] = model.predict(x_train, verbose=1)
    # params[model_chosen]['MSE_train'] = mean_squared_error(y_train, params[model_chosen]['pred_train'])
    
    # params[model_chosen]['pred_test'] = model.predict(x_test, verbose=1) 
    # params[model_chosen]['MSE_test'] = mean_squared_error(y_test, params[model_chosen]['pred_test'])

    # print('training set:', len(y_train))
    # print('testing set:', len(y_test))

    # # Upper limits for `l` & `u` in the cells below:

    # print(params.keys())

    # # Set `compare` in the cells below to a list
    # # containing any subset of these:

    # max_pts = 10**4
    # compare = params.keys() # e.g. ['rnn', 'alpharnn'] or ['lstm']
    # l, u = (None, None) # lower and upper indices of range to plot 
    # ds = max(1, len(y_train[l:u])//max_pts) # Downsampling ratio for under `max_pts`
    #                                         # per series.  Set `None` to disable. 

    # fig = plt.figure(figsize=(15,8))
    # x_vals = y_train_timestamps[l:u:ds]
    # for key in compare:
    #     y_vals = params[key]['pred_train'][l:u:ds]
    #     label = params[key]['label'] + ' (train MSE: %.2e)' % params[key]['MSE_train']
    #     plt.plot(x_vals, y_vals, c=params[key]['color'], label=label, lw=1)
    # plt.plot(x_vals, y_train[l:u:ds], c="black", label="Observed", lw=1)
    # start, end = x_vals.min(), x_vals.max()
    # xticks =  [start.date() + timedelta(days=(1+i)) for i in range(1 + (end - start).days)]
    # xticks = xticks[::max(1, len(xticks)//30)]
    # for t in xticks: plt.axvline(x=t, c='gray', linewidth=0.5, zorder=0)
    # plt.xticks(xticks, rotation=70)
    # plt.xlim(start, end)
    # plt.ylabel('$\hat{Y}$', rotation=0, fontsize=14)
    # plt.legend(loc="best", fontsize=12)
    # plt.title('Observed vs Model Outputs (Training)', fontsize=16)

    # compare = params.keys() # e.g. ['rnn', 'alpharnn'] or ['lstm']
    # l, u = (None, None) # e.g. (None, 100000) lower and upper indices of range to plot 
    # ds = max(1, len(y_train[l:u])//max_pts) # Downsampling ratio for under `max_pts`
    #                                         # per series.  Set `None` to disable.
    # fig = plt.figure(figsize=(15,8))
    # x_vals = y_test_timestamps[l:u:ds]
    # for key in compare:
    #     y_vals = params[key]['pred_test'][l:u:ds] - y_test[l:u:ds]
    #     label = params[key]['label'] + ' (test MSE: %.2e)' % params[key]['MSE_test']
    #     plt.plot(x_vals, y_vals, c=params[key]['color'], label=label, lw=1)
    # start, end = x_vals.min(), x_vals.max()
    # xticks =  [start.date() + timedelta(days=(1+i)) for i in range(1 + (end - start).days)]
    # xticks = xticks[::max(1, len(xticks)//30)]
    # plt.axhline(0, linewidth=0.8)
    # for t in xticks: plt.axvline(x=t, c='gray', linewidth=0.5, zorder=0)
    # plt.xticks(xticks, rotation=80)
    # plt.xlim(start, end)
    # plt.ylabel('$\hat{Y}-Y$', fontsize=14)
    # plt.legend(loc="best", fontsize=12)
    # plt.title('Observed vs Model Error (Testing)', fontsize=16)


    # compare = params.keys() # e.g. ['rnn', 'alpharnn'] or ['lstm']
    # l, u = (None, None) # lower and upper indices of range to plot - e.g. (None, 10000)
    # ds = max(1, len(y_train[l:u])//max_pts) # Downsampling ratio for under `max_pts`
    #                                         # per series.  Set `None` to disable.
    # fig = plt.figure(figsize=(15,8))
    # x_vals = y_train_timestamps[l:u:ds]
    # for key in compare:
    #     y_vals = params[key]['pred_train'][l:u:ds] - y_train[l:u:ds]
    #     label = params[key]['label'] + ' (train MSE: %.2e)' % params[key]['MSE_train']
    #     plt.plot(x_vals, y_vals, c=params[key]['color'], label=label, lw=1)
    # start, end = x_vals.min(), x_vals.max()
    # xticks =  [start.date() + timedelta(days=(1+i)) for i in range(1 + (end - start).days)]
    # xticks = xticks[::max(1, len(xticks)//30)]
    # plt.axhline(0, linewidth=0.8)
    # for t in xticks: plt.axvline(x=t, c='gray', linewidth=0.5, zorder=0)
    # plt.xticks(xticks, rotation=80)
    # plt.xlim(start, end)
    # plt.ylabel('$\hat{Y}-Y$', fontsize=14)
    # plt.legend(loc="best", fontsize=12)
    # plt.title('Observed vs Model Error (Training)', fontsize=16)

    # # number of samples to use for computing test statistic
    # n = 100000

    # params.keys()

    # predicted = params[model_chosen]['pred_test']
    # residual = y_test[-n:] - predicted[-n:]

    # lb, p = sm.stats.diagnostic.acorr_ljungbox(residual, lags=20, boxpierce=False)

    return

# def technical_indicators(df, train_date='2022-12-31'):
#     stats = {}

#     adf, p, usedlag, nobs, cvs, aic = sm.tsa.stattools.adfuller(df['close'])
#     adf_results_string = 'ADF: {}\np-value: {},\nN: {}, \ncritical values: {}'.format(adf, p, nobs, cvs)

#     pacf = sm.tsa.stattools.pacf(df['close'], nlags=30)
#     T = len(df['close'])

#     sig_test = lambda tau_h: np.abs(tau_h) > 2.58/np.sqrt(T)

#     n_steps = None
#     for i in range(len(pacf)):
#         if sig_test(pacf[i]) == False:
#             n_steps = i - 1
#             break

#     pacf_results_string = 'PACF n_steps: {}'.format(n_steps)
#     return adf_results_string + '\n' + pacf_results_string



#     df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

#     df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

#     df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
                                                            
#     df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
                                                            
#     df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])

#     def compute_atr(stock_data):
#         atr = pandas_ta.atr(high=stock_data['high'],
#                             low=stock_data['low'],
#                             close=stock_data['close'],
#                             length=14)
#         return atr.sub(atr.mean()).div(atr.std())

#     df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

#     def compute_macd(close):
#         macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
#         return macd.sub(macd.mean()).div(macd.std())

#     df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

#     df['dollar_volume'] = (df['adj close']*df['volume'])/1e6    

#     stats_results = stock_data.get_stats_tests()
#     return render_template('result.html', stats_results=stats_results)



