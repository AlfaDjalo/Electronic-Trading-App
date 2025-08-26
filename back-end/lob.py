# import requests

# def get_historical_prices(symbol="BTCUSDT", interval="1m", limit=100):
#     url = "https://api.binance.com/api/v3/klines"
#     params = {
#         "symbol": symbol,
#         "interval": interval,  # Options: "1m", "5m", "15m", "1h", etc.
#         "limit": limit  # Number of data points (max 1000)
#     }
    
#     response = requests.get(url, params=params)
#     data = response.json()
    
#     # Format: [Open time, Open, High, Low, Close, Volume, ...]
#     prices = [{"timestamp": x[0], "open": x[1], "high": x[2], "low": x[3], "close": x[4], "volume": x[5]} for x in data]
    
#     return prices

# # Fetch and print last 100 one-minute bars
# historical_prices = get_historical_prices()
# print(historical_prices[:5])  # Print first 5 results


# import pandas as pd
# print(pd.__version__)
# df = pd.DataFrame({"A": [1, 2, 3]})
# print(df)



# import requests
# import pandas as pd

# def get_order_book(symbol="BTCUSDT", depth=10):
#     """Fetches the latest order book snapshot from Binance."""
#     url = f"https://api.binance.com/api/v3/depth"
#     params = {"symbol": symbol, "limit": depth}
    
#     response = requests.get(url, params=params)
#     data = response.json()
    
#     bids = pd.DataFrame(data["bids"], columns=["price", "size"]).astype(float)
#     asks = pd.DataFrame(data["asks"], columns=["price", "size"]).astype(float)
    
#     return bids, asks

# # Fetch order book data
# bids, asks = get_order_book()

# print("Bids (Buy Orders):\n", bids.head(20))
# print("Asks (Sell Orders):\n", asks.head(20))


import requests
import pandas as pd

def get_historical_order_book(symbol="BTCUSDT", depth=100):
    url = f"https://api.binance.com/api/v3/depth"
    params = {"symbol": symbol, "limit": depth}
    response = requests.get(url, params=params)
    data = response.json()
    
    # Convert to DataFrame
    bids = pd.DataFrame(data["bids"], columns=["price", "size"]).astype(float)
    asks = pd.DataFrame(data["asks"], columns=["price", "size"]).astype(float)
    
    return bids, asks

# Get data
bids, asks = get_historical_order_book()
print("Bids:\n", bids.head())
print("Asks:\n", asks.head())