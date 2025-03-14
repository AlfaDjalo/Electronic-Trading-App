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

    # The mean squared error
    print('Root Mean Squared Error: {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))

    # Explained variance score: 1 is perfect prediction
    print('Variance Score: {0:.2f}'.format(r2_score(y_test, y_pred)))

    plt.scatter(y_test, y_pred)
    plt.plot([5, 15], [5, 15], 'r--', label='perfect fit')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()

    print('Root Mean Squared Error: {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, X_test.min_1_close))))

    return plt




def regression_on_trend(df, ):
    # Compute n-day lag close price within each ticker group
    df['min_1_close'] = df.groupby('ticker')['close'].shift(1)
    df['min_2_close'] = df.groupby('ticker')['close'].shift(2)
    df['min_3_close'] = df.groupby('ticker')['close'].shift(3)
    df['min_4_close'] = df.groupby('ticker')['close'].shift(4)

    # Compute log returns
    df['log_ret'] = np.log(df['close']) - np.log(df['min_1_close'])

    df['variance'] = df['log_ret'].rolling(180).var()

    df['min_1_trend'] = np.where(df['close'] > df['min_1_close'], 1, -1)
    df['min_2_trend'] = np.where(df['min_1_close'] > df['min_2_close'], 1, -1)
    df['min_3_trend'] = np.where(df['min_2_close'] > df['min_3_close'], 1, -1)

    df['trend_3_day'] = np.where(df['min_1_trend']+df['min_2_trend']+df['min_3_trend'] > 0, 1, -1)

    # start_date = pd.to_datetime('2018-06-01')
    # end_date = pd.to_datetime('2018-07-31')

    # Select a specific ticker (e.g., 'AAPL')
    # ticker = 'AGL.AX'  # Change this to the ticker you want to plot
    # df_ticker = df.xs(ticker, level='ticker')  # Extract data for the ticker

    # Filter by date range
#     filtered_df = df_ticker.loc[
#         (df_ticker.index.get_level_values('date') >= start_date) &
#         (df_ticker.index.get_level_values('date') <= end_date)
#     ]

# start_date = pd.to_datetime('2018-06-01')
# end_date = pd.to_datetime('2018-07-31')

# Select a specific ticker (e.g., 'AAPL')
# ticker = 'AGL.AX'  # Change this to the ticker you want to plot
# df_ticker = df.xs(ticker, level='ticker')  # Extract data for the ticker

# Filter by date range
# filtered_df = df_ticker.loc[
#     (df_ticker.index.get_level_values('date') >= start_date) &
#     (df_ticker.index.get_level_values('date') <= end_date)
# ]

    # Create the plot
    # plt.figure(figsize=(12, 6))

    # # Plot the closing price
    # plt.plot(
    #     filtered_df.index, filtered_df['close'], 'k--', label='Close Price'
    # )

    # # Scatter plot for positive trend
    # plt.scatter(
    #     filtered_df.index[filtered_df['trend_3_day'] == 1],
    #     filtered_df['close'][filtered_df['trend_3_day'] == 1],
    #     color='b', label='Pos Trend'
    # )

    # # Scatter plot for negative trend
    # plt.scatter(
    #     filtered_df.index[filtered_df['trend_3_day'] == -1],
    #     filtered_df['close'][filtered_df['trend_3_day'] == -1],
    #     color='r', label='Neg Trend'
    # )

    # Formatting
    # plt.xlabel("Date")
    # plt.ylabel("Close Price")
    # plt.title(f"{ticker} Closing Prices & Trends")
    # plt.legend()
    # plt.xticks(rotation=90)
    # plt.grid()

    # plt.show()

    # ticker = 'AGL.AX'  # Replace with the ticker you want
    # train_date = '2022-12-31'

    features = ['min_1_close', 'trend_3_day']
    target = 'close'

    # Filter data for the specific ticker
    df_ticker = df.loc[df.index.get_level_values('ticker') == ticker]

    # Split the data based on the date
    X_train = df_ticker.loc[df_ticker.index.get_level_values('date') <= train_date, features]
    X_test = df_ticker.loc[df_ticker.index.get_level_values('date') > train_date, features]

    y_train = df_ticker.loc[df_ticker.index.get_level_values('date') <= train_date, target]
    y_test = df_ticker.loc[df_ticker.index.get_level_values('date') > train_date, target]

    # Create linear regression object
    # regr = linear_model.LinearRegression(fit_intercept=False)
    regr = linear_model.LinearRegression(fit_intercept=True)

    # Train the model using the training set
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    # The mean squared error
    print('Root Mean Squared Error: {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))

    # Explained variance score: 1 is perfect prediction
    print('Variance Score: {0:.2f}'.format(r2_score(y_test, y_pred)))

    plt.scatter(y_test, y_pred)
    plt.plot([5, 15], [5, 15], 'r--', label='perfect fit')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()

    print('Root Mean Squared Error: {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, X_test.min_1_close))))

    return plt



