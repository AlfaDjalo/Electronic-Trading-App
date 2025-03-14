from flask import render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from StockData import StockData
import matplotlib.pyplot as plt
import io
import base64

def setup_routes(app):
    @app.route("/")
    def index():
        asx200 = pd.read_html('https://en.wikipedia.org/wiki/S%26P/ASX_200')[2]
        tickers = asx200[['Code', 'Company']].to_dict(orient="records")
        return render_template("index.html", tickers=tickers)

    @app.route('/load_data', methods=['POST'])
    def load_data():
        selected_ticker = request.form.get('ticker')
        selected_ticker +=  '.AX'        
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        stock_data = StockData(selected_ticker, start_date, end_date)
        return redirect(url_for('select_algorithm'))

    @app.route('/select_algorithm')
    def select_algorithm():
        return render_template('select_algorithm.html')

    @app.route('/run_algorithm', methods=['POST'])
    def run_algorithm():
        algorithm = request.form['algorithm']
        if algorithm == 'regression':
            df = stock_data.get_data()
            df['min_1_close'] = df['close'].shift(1)
            df.dropna(inplace=True)
            X = df[['min_1_close']]
            y = df['close']
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            df['predicted_close'] = model.predict(X)
            
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df['close'], label='Actual Close')
            plt.plot(df.index, df['predicted_close'], label='Predicted Close')
            plt.legend()
            plt.title(f'{selected_ticker} Close Price Prediction')
            
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            return render_template('result.html', plot_url=plot_url)
        return "Algorithm not implemented", 400