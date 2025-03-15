import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from arch import arch_model

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



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



    # # Create linear regression object
    # # regr = linear_model.LinearRegression(fit_intercept=False)
    # regr = linear_model.LinearRegression(fit_intercept=True)

    # # Train the model using the training set
    # regr.fit(X_train, y_train)

    # # Make predictions using the testing set
    # y_pred = regr.predict(X_test)

    # # The mean squared error
    # print('Root Mean Squared Error: {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))

    # # Explained variance score: 1 is perfect prediction
    # print('Variance Score: {0:.2f}'.format(r2_score(y_test, y_pred)))

    # plt.scatter(y_test, y_pred)
    # plt.plot([5, 15], [5, 15], 'r--', label='perfect fit')
    # plt.xlabel('Actual')
    # plt.ylabel('Predicted')
    # plt.legend()

    # print('Root Mean Squared Error: {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, X_test.min_1_close))))

    # return plt



