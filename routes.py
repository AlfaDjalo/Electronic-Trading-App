from flask import render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
from StockData import StockData
import matplotlib.pyplot as plt
import io
import base64
from models import ModelHandler
# from models import regression, regression_on_trend, ML_regression, ModelHandler

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
        model_handler = ModelHandler(stock_data.get_data(), train_date='2022-12-31')

        algorithm = request.form['algorithm']
        if algorithm == 'regression':
            model_handler.regression()
            stats = model_handler.get_stats()
            plt = model_handler.get_plt()
            # stats, plt = model_handler.regression()
        elif algorithm == 'regression_on_trend':
            # stats, plt = model_handler.regression_on_trend()
            model_handler.regression_on_trend()
            stats = model_handler.get_stats()
            plt = model_handler.get_plt()
        elif algorithm == 'ml_regression':
            # stats, plt = model_handler.ml_regression(do_training=False)
            model_handler.ml_regression(do_training=False)
            stats = model_handler.get_stats()
            plt = model_handler.get_plt()
        else:
            return "Algorithm not implemented", 400

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return render_template('result.html', plot_url=plot_url, stats=stats)