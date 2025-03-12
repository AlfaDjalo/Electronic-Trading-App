from flask import render_template, request
import pandas as pd
import numpy as np

from StockData import StockData

def setup_routes(app):
    @app.route("/")
    def index():

        asx200 = pd.read_html('https://en.wikipedia.org/wiki/S%26P/ASX_200')[2]

    # Rename columns to match expected values
        # asx200.rename(columns={'Ticker': 'Code', 'Company Name': 'Company'}, inplace=True)

        # Convert DataFrame to a list of dictionaries
        tickers = asx200[['Code', 'Company']].dropna().to_dict(orient='records')

        return render_template("index.html", tickers=tickers)


    @app.route('/load_data', methods=['POST'])
    def load_data():
        # Get form data
        selected_ticker = request.form.get('ticker')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        # Print for debugging
        print(f"Selected Ticker: {selected_ticker}, Start Date: {start_date}, End Date: {end_date}")

        # Run your existing code here using selected_ticker, start_date, and end_date
        # Example: (Replace this with your actual logic)
        stock = StockData(selected_ticker, start_date, end_date)
        
        asx200 = pd.read_html('https://en.wikipedia.org/wiki/S%26P/ASX_200')[2]

    # Rename columns to match expected values
        # asx200.rename(columns={'Ticker': 'Code', 'Company Name': 'Company'}, inplace=True)

        # Convert DataFrame to a list of dictionaries
        tickers = asx200[['Code', 'Company']].dropna().to_dict(orient='records')

        # Return the result (modify as needed)
        return render_template("index.html", tickers=tickers)