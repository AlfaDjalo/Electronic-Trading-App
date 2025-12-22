# first install the package if you haven't
# pip install nasdaq-data-link

import nasdaqdatalink
import pandas as pd

# Optionally: set your API key explicitly — or export as env var: NASDAQ_DATA_LINK_API_KEY

def fetch_price_history(dataset_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetch historical time‑series data for a given Nasdaq Data Link dataset.

    Args:
      dataset_code: e.g. 'XNAS/AAPL' or another code from Nasdaq Data Link catalog.
      start_date: 'YYYY-MM-DD' or None
      end_date: 'YYYY-MM-DD' or None

    Returns:
      pandas DataFrame with date as index and columns per the dataset (e.g. Open, High, Low, Close, Volume, etc.)
    """
    data = nasdaqdatalink.get(dataset_code, start_date=start_date, end_date=end_date)
    # Often the returned DataFrame has a 'Date' column — convert to index
    print(data)
    
    if 'Date' in data.columns:
        data = data.set_index('Date')
    return data

if __name__ == "__main__":
    # Example: fetch Apple historic prices from 2020‑01‑01 to 2021‑01‑01
    nasdaqdatalink.ApiConfig.api_key = '2g89y4rP1YSsNETSYj7k'
    
    print(nasdaqdatalink.ApiConfig.api_key)

    meta = nasdaqdatalink.get_table("ZACKS/FC")
    print(meta.head())

    df = fetch_price_history("FRED/GDP", start_date="2020-01-01", end_date="2021-01-01")
    print(df.head())
