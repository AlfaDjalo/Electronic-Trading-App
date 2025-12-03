import numpy as np

def process_backtests(predictions, backtest_list):
    """
    Loops through backtests.
    """
    results = {}

    for bt in backtest_list:
        model_name = bt["modelName"]
        params = bt["backtestParams"]

        series = prepare_series(predictions, model_name)        
        
        bt_result = apply_strategy(
            preds = series["preds"],
            actuals = series["actual"],
            dates = series["dates"],
            params = params
        )

        results[bt["id"]] = bt_result

    return results


def prepare_series(predictions, model_name):
    """
    Extracts array or predicted values, actual pries, dates.
    """
    preds_list = []
    actuals_list = []
    dates_list = predictions["dates"][model_name]
    
    for i, date in enumerate(dates_list):
        actuals_list.append(predictions["actual"][date][0][0])
        preds_list.append(predictions["predictions"][model_name][i][0])

    preds_array = np.array(preds_list)
    actual_array = np.array(actuals_list)

    return {
        "preds": preds_array,
        "actual": actual_array,
        "dates": dates_list
    }

def apply_strategy(preds, actuals, dates, params):
    """
    Contains the trading logic.
    """
    cash = params["initialCash"]
    execution_delay = params["delay"]
    transaction_cost = params["transactionCost"]
    # slippage = params["slippage"]

    position = 0
    entry_price = None
    stopped_out = False
    time_out = 0

    equity = []
    trades = []
    
    print("Starting backtest.")

    # Initial buy
    # price0 = actuals[0]
    # entry_price = price0
    # position = cash / entry_price
    # cash = cash - position * entry_price

    # trades.append({
    #     "date": dates[0],
    #     "buy/sell:": "buy",
    #     "shares": position,
    #     "price": price0
    # })

    # print("Entering position.")

    # cash_series = [cash]
    # position_series = [position]

    for t in range(len(dates)):
        time_out +=1

        price = actuals[t]
        prediction = preds[t]

        portfolio_value = cash + position * price
        
        print("Position: ", position, ", price: ", price, ", prediction: ", prediction, "portfolio value: ", portfolio_value)

        equity.append({
            "date": dates[t],
            "value": portfolio_value
        })

        # total_return = equity[-1] / equity[0] - 1
        max_drawdown = 0
        # position_series = 1
        # cash_series = 1
        
        if stopped_out:
            continue

        if position == 0:
            if time_out > execution_delay:
                if check_entry(price, prediction):
                    position = cash / (price * (1 + transaction_cost))
                    cash = 0
                    entry_price = price
                    trades.append({
                        "date": dates[t],
                        "buy/sell:": "buy",
                        "shares": position,
                        "price": price
                    })
                    print("Entering position, price: ", price, ", prediction: ", prediction)
                    continue                
        else:
            if(check_stop_loss(position, entry_price, price, params)):
                trades.append({
                    "date": dates[t],
                    "buy/sell:": "sell",
                    "shares": position,
                    "price": price
                })
                cash += position * (price * (1 - transaction_cost))
                position = 0
                stopped_out = True
                print("Stopped out.")
                continue
            elif (check_take_profit(position, entry_price, price, params)):
                trades.append({
                    "date": dates[t],
                    "buy/sell:": "sell",
                    "shares": position,
                    "price": price
                })
                cash += position * (price * (1 - transaction_cost))
                position = 0
                time_out = 0
                print("Taking profit.")
                continue
            elif (check_exit(price, prediction)):
                trades.append({
                    "date": dates[t],
                    "buy/sell:": "sell",
                    "shares": position,
                    "price": price
                })
                cash += position * (price * (1 - transaction_cost))
                position = 0
                time_out = 0
                print("Exiting position, price: ", price, ", prediction: ", prediction)
                continue                





    print("Ending backtest.")

    # T = len(dates)
    price = actuals[-1]
    
    trades.append({
        "date": dates[-1],
        "buy/sell:": "sell",
        "shares": position,
        "price": price
    })
    cash += position * (price * (1 - transaction_cost))
    position = 0

    portfolio_value = cash + position * price
    
    equity.append({
        "date": dates[-1],
        "value": portfolio_value
    })

    stats = calculate_stats(equity, trades)

    print("trades: ", trades)

    return {
       "equity": equity,
       "trades": trades,
       "stats": stats,
       "model_name": params.get("modelName")
    }


def calculate_stats(equity_curve, trades):
    """
    Calculates stats.
    """
    
    return {
       "total_return": 1,
       "max_drawdown": -1,
       "sharpe": 0.5,
       "win_rate": 0.6
    }

def check_entry(price, prediction):
    if prediction > price:
        return True
    else:
        return False 

def check_stop_loss(position, entry_price, price, params):
    
    initial_cash = params["initialCash"]
    stop_loss = params["stopLoss"]

    # print("price: ", price, " stop loss: ", (1 + stop_loss / 100) * entry_price)

    if price < (1 + stop_loss / 100) * entry_price:
        return True
    else:
        return False

def check_take_profit(position, entry_price, price, params):
    # initial_cash = params["initialCash"]
    take_profit = params["takeProfit"]

    # print("price: ", price, " take profit: ",(1 + take_profit / 100)* entry_price)

    if price > (1 + take_profit / 100) * entry_price:
        return True
    else:
        return False


def check_exit(price, prediction):
    if prediction < price:
        return True
    else:
        return False 
