from flask import render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
from StockData import StockData
import matplotlib.pyplot as plt
import io
import base64
from models import regression, regression_on_trend

def setup_routes(app):
    app.secret_key = 'your_secret_key'  # Add a secret key for session management
    
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

        # Store data in session
        session['selected_ticker'] = selected_ticker
        session['start_date'] = start_date
        session['end_date'] = end_date

        return redirect(url_for('select_algorithm'))

    @app.route('/select_algorithm')
    def select_algorithm():
        return render_template('select_algorithm.html')

    @app.route('/run_algorithm', methods=['POST'])
    def run_algorithm():
        selected_ticker = session.get('selected_ticker')
        start_date = session.get('start_date')
        end_date = session.get('end_date')

        stock_data = StockData(selected_ticker, start_date, end_date)

        algorithm = request.form['algorithm']
        if algorithm == 'regression':
            stats, plt = regression(stock_data.get_data(), '2022-12-31')
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
  
            stats = {key: float(value) if isinstance(value, np.float64) else value for key, value in stats.items()}

            print(f"Stats type before passing to template: {type(stats)}")
            print(f"Stats content before passing to template: {stats}")

            if not isinstance(stats, dict):
                print("ERROR: stats is not a dictionary!", type(stats))

            return render_template('result.html', plot_url=plot_url, stats=stats)
        elif algorithm == 'regression_on_trend':   
            stats, plt = regression_on_trend(stock_data.get_data(), '2022-12-31')
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
  
            stats = {key: float(value) if isinstance(value, np.float64) else value for key, value in stats.items()}

            print(f"Stats type before passing to template: {type(stats)}")
            print(f"Stats content before passing to template: {stats}")

            if not isinstance(stats, dict):
                print("ERROR: stats is not a dictionary!", type(stats))

            return render_template('result.html', plot_url=plot_url, stats=stats)
        elif algorithm == 'banana':
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
        elif algorithm == 'stats_tests':
            stats_results = stock_data.get_stats_tests()
            return render_template('result.html', stats_results=stats_results)
        return "Algorithm not implemented", 400