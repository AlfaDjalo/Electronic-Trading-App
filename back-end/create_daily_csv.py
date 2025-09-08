#!/usr/bin/env python3
import argparse
import pandas as pd
import yfinance as yf
import sys

def download_stock_data(ticker, start_date, end_date, output_file):
    """
    Download daily stock data from Yahoo Finance and save as CSV.
    """
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")

    try:
        df = yf.download(tickers=ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)

        if df.empty:
            print(f"⚠️ No data found for {ticker} in the given date range.")
            return False

        # Ensure index is named and reset for CSV export
        df.index.name = "date"
        df = df.reset_index()

        # Flatten MultiIndex columns if necessary
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(col).strip().lower() for col in df.columns.values]
        else:
            df.columns = df.columns.str.lower()

        # # Normalize column names
        # df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"✅ Data saved to {output_file}")
        return True

    except Exception as e:
        print(f"❌ Error downloading {ticker}: {e}", file=sys.stderr)
        return False


def main():
    print("Starting.")
    parser = argparse.ArgumentParser(description="Download Yahoo Finance daily stock data to CSV")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL, TSLA)")
    parser.add_argument("start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "-o", "--output",
        help="Output CSV file name (default: <ticker>_<start>_<end>.csv)"
    )

    print("Parser intiated.")
    args = parser.parse_args()
    print("Args created.")
    output_file = args.output or f"{args.ticker}_{args.start}_{args.end}.csv"
    print(f"Output file: {output_file}")
    download_stock_data(args.ticker, args.start, args.end, output_file)
    print("download_stock_data run")

if __name__ == "__main__":
    main()
