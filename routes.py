from flask import render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import numpy as np
from StockData import StockData

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import io
import base64
from models import ModelHandler
import json
import os

def setup_routes(app):
    app.secret_key = 'your_secret_key'  # Add a secret key for session management

    @app.route("/")
    def index():
        # Initialize the session variable as a list if it doesn't exist
        if 'comparisons' not in session:
            print("Adding comparisons list to session")
            session['comparisons'] = []
            print(session.get('comparisons', []))
        
        asx200 = pd.read_html('https://en.wikipedia.org/wiki/S%26P/ASX_200')[2]
        tickers = asx200[['Code', 'Company']].to_dict(orient="records")
        comparisons = session.get('comparisons', [])  # Retrieve the list of comparisons
        return render_template("index.html", tickers=tickers, comparisons=comparisons)

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

        # session['stock_data'] = StockData(selected_ticker, start_date, end_date)

        # print(session['stock_data'].get_ticker())
        # print(session['stock_data'].get_dates())
        # print(session['stock_data'].get_dates()["start_date"])
        # print(session['stock_data'].get_dates()["end_date"])

        return redirect(url_for('select_algorithm'))

    @app.route('/select_algorithm')
    def select_algorithm():
        return render_template('select_algorithm.html')

    @app.route('/run_algorithm', methods=['POST'])
    def run_algorithm():
        selected_ticker = session.get('selected_ticker')
        start_date = session.get('start_date')
        end_date = session.get('end_date')

        stock_data = StockData(selected_ticker, start_date, end_date, load_data=True, create_model_data=True)
        model_handler = ModelHandler(stock_data.get_data(), train_date='2022-12-31')

        algorithm = request.form['algorithm']
        if algorithm == 'regression':
            model_handler.regression()
            stats = model_handler.get_stats()
            plt = model_handler.get_plt()
            # fig = model_handler.get_fig()
            # stats, plt = model_handler.regression()
        elif algorithm == 'regression_on_trend':
            # stats, plt = model_handler.regression_on_trend()
            model_handler.regression_on_trend()
            stats = model_handler.get_stats()
            plt = model_handler.get_plt()
        elif algorithm == 'ml_regression':
            model_handler.ml_regression(do_training=True)
            stats = model_handler.get_stats()
            plt = model_handler.get_plt()
        else:
            return "Algorithm not implemented", 400
        # plt.savefig(img, format='png')
        # img.seek(0)
        # plot_url = base64.b64encode(img.getvalue()).decode()
        # plt.close()

        plt.close('all')
        # plt.close(fig)
        # plt.close(fig)  # Close just this figure
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('result.html', plot_url=plot_url, stats=stats)

    @app.route('/create_comparison', methods=['POST'])
    def create_comparison():
        ticker = request.form.get('ticker') + '.AX'
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        model = request.form.get('model')

        # Create a new comparison dictionary
        comparison = {
            'ticker': ticker,
            'model': model,
            'start_date': start_date,
            'end_date': end_date
        }

        # Retrieve the list of comparisons from the session
        comparisons = session.get('comparisons', [])
        comparisons.append(comparison)  # Add the dictionary directly
        session['comparisons'] = comparisons  # Save the updated list back to the session
        print("Comparisons:")
        print(session['comparisons'])

        return redirect(url_for('index'))

    @app.route('/get_comparisons')
    def get_comparisons():
        # Retrieve the list of comparisons from the session
        comparisons = session.get('comparisons', [])
        return render_template('comparisons.html', comparisons=comparisons)

    @app.route('/set_parameters/<int:index>', methods=['GET', 'POST'])
    def set_parameters(index):
        comparisons = session.get('comparisons', [])
        if index >= len(comparisons):
            return "Comparison not found", 404

        if request.method == 'POST':
            parameters = request.form.to_dict()
            # Save parameters to JSON file
            with open('model_parameters.json', 'r+') as f:
                all_parameters = json.load(f)
                all_parameters[comparisons[index]['model']] = parameters
                f.seek(0)
                json.dump(all_parameters, f, indent=4)
                f.truncate()
            comparisons[index]['parameters'] = parameters
            session['comparisons'] = comparisons
            return redirect(url_for('index'))

        model = comparisons[index]['model']
        with open('model_parameters.json') as f:
            all_parameters = json.load(f)
        model_parameters = all_parameters.get(model, {})
        common_parameters = all_parameters.get('common', {})
        return render_template(
            'set_parameters.html',
            model=model,
            model_parameters=model_parameters,
            common_parameters=common_parameters,
            comparison=comparisons[index]
        )

    @app.route('/edit_comparison/<int:index>', methods=['GET', 'POST'])
    def edit_comparison(index):
        comparisons = session.get('comparisons', [])
        if index >= len(comparisons):
            return "Comparison not found", 404

        if request.method == 'POST':
            comparisons[index]['ticker'] = request.form.get('ticker') + '.AX'
            comparisons[index]['start_date'] = request.form.get('start_date')
            comparisons[index]['end_date'] = request.form.get('end_date')
            comparisons[index]['model'] = request.form.get('model')
            session['comparisons'] = comparisons
            return redirect(url_for('index'))

        return render_template('edit_comparison.html', comparison=comparisons[index])

    @app.route('/delete_comparison/<int:index>', methods=['POST'])
    def delete_comparison(index):
        comparisons = session.get('comparisons', [])
        if index >= len(comparisons):
            return "Comparison not found", 404

        comparisons.pop(index)
        session['comparisons'] = comparisons
        return redirect(url_for('index'))

    @app.route('/clear_comparisons', methods=['POST'])
    def clear_comparisons():
        session['comparisons'] = []
        return redirect(url_for('index'))

    @app.route('/run_comparisons', methods=['POST'])
    def run_comparisons():
        comparisons = session.get('comparisons', [])
        results = []

        for comparison in comparisons:
            ticker = comparison['ticker']
            start_date = comparison['start_date']
            end_date = comparison['end_date']
            model = comparison['model']

            stock_data = StockData(ticker, start_date, end_date, load_data=True, create_model_data=True)
            model_handler = ModelHandler(stock_data.get_data(), train_date='2022-12-31')

            if model == 'LinearRegression':
                model_handler.regression()
            elif model == 'LSTM':
                model_handler.ml_regression(do_training=True)
            elif model == 'RNN':
                model_handler.ml_regression(do_training=True)  # Assuming RNN uses the same method
            else:
                results.append({'ticker': ticker, 'model': model, 'error': 'Model not implemented'})
                continue

            stats = model_handler.get_stats()
            results.append({'ticker': ticker, 'model': model, 'stats': stats})

        return render_template('comparison_results.html', results=results)