import websocket
import json
import time
import csv

# File to save data
csv_filename = "order_book_history.csv"

# Define WebSocket URL
ws_url = "wss://stream.binance.com:9443/ws/btcusdt@depth"

# Open CSV file and write headers
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", 
                     "bid_price_0", "bid_price_1", "bid_price_2", "bid_price_3", "bid_price_4",
                     "bid_volume_0", "bid_volume_1", "bid_volume_2", "bid_volume_3", "bid_volume_4",
                     "ask_price_0", "ask_price_1", "ask_price_2", "ask_price_3", "ask_price_4",
                     "ask_volume_0", "ask_volume_1", "ask_volume_2", "ask_volume_3", "ask_volume_4"])

# function to capture data one record for each timestamp
def on_message(ws, message):
    try:
        data = json.loads(message)
        # Check if 'b' (bids) and 'a' (asks) keys exist
        if 'b' not in data or 'a' not in data:
            print("Error: Missing 'b' or 'a' in message")
            return
        
        # Get current timestamp
        timestamp = int(time.time())

        # Prepare row data
        bid_prices = [data['b'][i][0] for i in range(5)]  # Top 5 bid prices
        bid_volumes = [data['b'][i][1] for i in range(5)]  # Top 5 bid volumes
        ask_prices = [data['a'][i][0] for i in range(5)]  # Top 5 ask prices
        ask_volumes = [data['a'][i][1] for i in range(5)]  # Top 5 ask volumes

        row = [timestamp] + bid_prices + bid_volumes + ask_prices + ask_volumes

        # Save to CSV file
        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)

        print(f"Saved: {timestamp}")
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"Error processing message: {e}")

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket Closed")

def on_open(ws):
    print("WebSocket Connection Opened")

# Start WebSocket connection
ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_close=on_close)
ws.on_open = on_open
ws.run_forever()
