""" Manage routes """

# import io
import base64
import json
# import os

from flask import render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

from models import ModelHandler
from stock_data import StockData

def create_prediction_chart(x_test_index, y_test, predictions):
    """Generate a chart comparing actual values and multiple predicted series."""
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(x_test_index, y_test, label="Actual", linestyle='dashed')
    for model_name, y_pred in predictions.items():
        ax.plot(x_test_index, y_pred, label=f"Predicted ({model_name})")
    ax.set_title("Predictions vs Actual Values")
    ax.set_xlabel("Date")
    ax.set_ylabel("Values")
    ax.legend()
    plt.xticks(rotation=45)
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return chart_url

def create_error_chart(x_test_index, errors):
    """Generate a chart showing the error series for all models."""
    print(errors)
    fig, ax = plt.subplots(figsize=(15, 8))
    for model_name, error in errors.items():
        ax.plot(x_test_index, error, label=f"Error ({model_name})")
    ax.set_title("Error Between Predictions and Actual Values")
    ax.set_xlabel("Date")
    ax.set_ylabel("Error")
    ax.legend()
    plt.xticks(rotation=45)
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    error_chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return error_chart_url

def setup_routes(app):
    """ Setup routes """
    app.secret_key = 'your_secret_key'  # Add a secret key for session management

    @app.route("/")
    def index():
        # Initialize the session variable as a list if it doesn't exist
        if 'comparisons' not in session:
            print("Adding comparisons list to session")
            session['comparisons'] = []
            print(session.get('comparisons', []))

        category = request.args.get('category', 'australian')  # Default to Australian stocks
        tickers = []

        if category == 'australian':
            asx200 = pd.read_html('https://en.wikipedia.org/wiki/S%26P/ASX_200')[2]
            tickers = asx200[['Code', 'Company']].to_dict(orient="records")
        elif category == 'us':
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
            # symbols_list = sp500['Symbol'].unique().tolist()
            tickers = sp500[['Symbol', 'Security']].rename(columns={'Symbol': 'Code', 'Security': 'Company'}).to_dict(orient="records")
        elif category == 'fx':
            tickers = [{'Code': 'AUDUSD=X', 'Company': 'AUDUSD'}]
        elif category == 'crypto':
            tickers = [{'Code': 'BTC-USD', 'Company': 'Bitcoin'}]

        comparisons = session.get('comparisons', [])  # Retrieve the list of comparisons
        return render_template("index.html", tickers=tickers, comparisons=comparisons, category=category)

    @app.route('/create_comparison', methods=['POST'])
    def create_comparison():
        category = request.form.get('category', 'default')  # Default to Australian stocks
        print("Category", category)
        ticker = request.form.get('ticker')
        if category == 'australian':
            ticker += '.AX'
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        model = request.form.get('model')

        # Load full parameter metadata from model_parameters.json
        with open('model_parameters.json') as f:
            all_parameters = json.load(f)
        model_parameters = all_parameters.get(model, {})

        # Initialize the value field for each parameter with its default value
        for param, metadata in model_parameters.items():
            metadata['value'] = metadata.get('default')

        # Create a new comparison dictionary
        comparison = {
            'ticker': ticker,
            'model': model,
            'start_date': start_date,
            'end_date': end_date,
            'params': model_parameters,  # Include full parameter metadata with initialized values
            'use_log_returns': False  # Default to off
        }

        # Retrieve the list of comparisons from the session
        comparisons = session.get('comparisons', [])
        comparisons.append(comparison)  # Add the dictionary directly
        session['comparisons'] = comparisons  # Save the updated list back to the session

        return redirect(url_for('index'))

    @app.route('/set_parameters/<int:index>', methods=['GET', 'POST'])
    def set_parameters(index):
        comparisons = session.get('comparisons', [])
        if index >= len(comparisons):
            return "Comparison not found", 404

        if request.method == 'POST':
            parameters = request.form.to_dict()
            model_metadata = comparisons[index]['params']
            print("Getting parameters")
            # Validate inputs and update the value field
            for param, value in parameters.items():
                metadata = model_metadata.get(param, {})
                if not isinstance(metadata, dict):
                    return f"Invalid metadata for parameter '{param}'", 400
                if metadata.get("type") == "integer":
                    value = int(value)
                    if value < metadata.get("min", float('-inf')) or value > metadata.get("max", float('inf')):
                        return f"Invalid value for {param}", 400
                elif metadata.get("type") == "float":
                    value = float(value)
                    if value < metadata.get("min", float('-inf')) or value > metadata.get("max", float('inf')):
                        return f"Invalid value for {param}", 400
                elif metadata.get("type") == "category":
                    if value not in metadata.get("values", []):
                        return f"Invalid value for {param}", 400
                elif metadata.get("type") == "boolean":
                    try:
                        value = bool(int(value)) if value else False  # Convert '0' or '1' to False or True
                    except ValueError:
                        return f"Invalid value for {param}: must be '0' or '1'", 400
                # Save validated value back to the metadata dictionary
                metadata['value'] = value

            use_log_returns = request.form.get('use_log_returns', 'off') == 'on'
            comparisons[index]['use_log_returns'] = use_log_returns

            # Save updated parameters to the comparison and session
            comparisons[index]['params'] = model_metadata
            session['comparisons'] = comparisons

            return redirect(url_for('index'))

        model = comparisons[index]['model']
        model_parameters = comparisons[index]['params']

        # Ensure metadata['values'] is a list
        for param, metadata in model_parameters.items():
            if metadata.get('type') == 'category' and callable(metadata.get('values')):
                metadata['values'] = list(metadata['values']())

        return render_template(
            'set_parameters.html',
            model=model,
            model_parameters=model_parameters,
            comparison=comparisons[index],
            use_log_returns=comparisons[index].get('use_log_returns', False)
        )

    @app.route('/edit_comparison/<int:index>', methods=['GET', 'POST'])
    def edit_comparison(index):
        print("In edit comparison")
        comparisons = session.get('comparisons', [])
        if index >= len(comparisons):
            return "Comparison not found", 404

        if request.method == 'POST':
            comparisons[index]['ticker'] = request.form.get('ticker') + '.AX' # Needs fixing
            comparisons[index]['start_date'] = request.form.get('start_date')
            comparisons[index]['end_date'] = request.form.get('end_date')
            comparisons[index]['model'] = request.form.get('model')
            session['comparisons'] = comparisons
            return redirect(url_for('index'))

        return render_template('edit_comparison.html', comparison=comparisons[index])

    @app.route('/delete_comparison/<int:index>', methods=['POST', 'GET'])
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

    @app.route('/run_comparisons', methods=['GET', 'POST'])
    def run_comparisons():
        comparisons = session.get('comparisons', [])
        if not comparisons:
            return render_template('comparison_results.html', results=[], unique_tickers=[], selected_ticker=None)

        selected_ticker = request.args.get('ticker', comparisons[0]['ticker'])
        filtered_comparisons = [c for c in comparisons if c['ticker'] == selected_ticker]

        results = []
        predictions = {}
        errors = {}
        y_test = None
        x_test_index = None

        # Handle GET requests (e.g., filtering by ticker)
        if request.method == 'GET':
            for comparison in filtered_comparisons:
                results.append({
                    'ticker': comparison['ticker'],
                    'model': comparison['model'],
                    'stats': 'N/A',
                    'error': 'N/A',
                })

            unique_tickers = sorted(set(c['ticker'] for c in comparisons))
            return render_template(
                'comparison_results.html',
                results=results,
                unique_tickers=unique_tickers,
                selected_ticker=selected_ticker,
                prediction_chart_url=None,
                error_chart_url=None
            )

        for comparison in filtered_comparisons:
            ticker = comparison['ticker']
            start_date = comparison['start_date']
            end_date = comparison['end_date']
            model = comparison['model']
            params = comparison.get('params', {})
            stock_data = StockData(ticker, start_date, end_date, load_data=True, create_model_data=True)
            features = stock_data.create_feature_list(comparison)
            target = stock_data.create_target(comparison)
            stock_data.split_data()
            if params.get('standardise', {}).get('value', False):
                stock_data.standardise_input(stock_data.features)
            ML_data = stock_data.get_data2()
            model_handler = ModelHandler(ML_data, params)

            try:
                if model == 'LinearRegression':
                    model_handler.regression()
                elif model == 'RNN':
                    model_handler.ML(model_handler.simpleRNN_)
                elif model == 'LSTM':
                    model_handler.ML(model_handler.lstm_)

                y_pred = model_handler.model.predict(ML_data['x_test'])
                if y_test is None:  # Ensure y_test is assigned only once
                    y_test = ML_data['y_test']
                    x_test_index = ML_data['x_test'].index

                print("Adding prediction for ", model)
                predictions[model] = y_pred  # Ensure predictions are flattened
                print("Adding error for ", model)
                errors[model] = (y_test - y_pred) if y_test is not None else None

                results.append({
                    'ticker': ticker,
                    'model': model,
                    'stats': model_handler.get_stats(),
                    'error': 'N/A'
                })
            except Exception as e:
                results.append({
                    'ticker': ticker,
                    'model': model,
                    'stats': 'N/A',
                    'error': str(e)
                })

        # Ensure x_test_index and y_test are not None before generating charts
        prediction_chart_url = None
        error_chart_url = None
        if x_test_index is not None and y_test is not None and predictions:
            prediction_chart_url = create_prediction_chart(x_test_index, y_test, predictions)
            error_chart_url = create_error_chart(x_test_index, errors)

        unique_tickers = sorted(set(c['ticker'] for c in comparisons))

        return render_template(
            'comparison_results.html',
            results=results,
            unique_tickers=unique_tickers,
            selected_ticker=selected_ticker,
            prediction_chart_url=prediction_chart_url,
            error_chart_url=error_chart_url
        )

    @app.route('/detailed_results/<int:index>')
    def detailed_results(index):
        comparisons = session.get('comparisons', [])
        if index >= len(comparisons):
            return "Comparison not found", 404

        comparison = comparisons[index]
        ticker = comparison['ticker']
        start_date = comparison['start_date']
        end_date = comparison['end_date']
        model = comparison['model']
        params = comparison.get('params', {})

        stock_data = StockData(ticker, start_date, end_date, load_data=True, create_model_data=True)
        features = stock_data.create_feature_list(comparison)
        target = stock_data.create_target(comparison)
        stock_data.split_data()
        stock_data.standardise_input(stock_data.features)
        ML_data = stock_data.get_data2()
        model_handler = ModelHandler(ML_data, params)

        try:
            if model == 'LinearRegression':
                model_handler.regression()
            elif model == 'RNN':
                model_handler.ML(model_handler.simpleRNN_)
            elif model == 'LSTM':
                model_handler.ML(model_handler.lstm_)
            else:
                return "Model not implemented", 400

            # Generate charts
            y_pred = model_handler.model.predict(ML_data['x_test'])
            x_test_index = ML_data['x_test'].index
            prediction_chart_url = create_prediction_chart(x_test_index, y_pred, ML_data['y_test'])
            error = np.abs(ML_data['y_test'] - y_pred.flatten())
            error_chart_url = create_error_chart(x_test_index, error)

            return render_template('detailed_results.html', comparison=comparison, prediction_chart_url=prediction_chart_url, error_chart_url=error_chart_url)

        except Exception as e:
            return f"Error generating detailed results: {str(e)}", 500

    @app.route('/comparison_results', methods=['GET'])
    def comparison_results():
        comparisons = session.get('comparisons', [])
        results = []

        # Filter by ticker if a ticker is selected
        selected_ticker = request.args.get('ticker', None)
        filtered_comparisons = [c for c in comparisons if not selected_ticker or c['ticker'] == selected_ticker]

        for comparison in filtered_comparisons:
            results.append({
                'ticker': comparison['ticker'],
                'model': comparison['model'],
                'stats': 'N/A',  # Placeholder for stats
                'error': 'N/A',  # Placeholder for error
            })

        # Generate charts for the selected ticker
        prediction_chart_url = None
        error_chart_url = None
        if selected_ticker:
            grouped_comparisons = {}
            for comparison in filtered_comparisons:
                key = (comparison['ticker'], comparison['start_date'], comparison['end_date'])
                grouped_comparisons.setdefault(key, []).append(comparison)

            for (ticker, start_date, end_date), group in grouped_comparisons.items():
                stock_data = StockData(ticker, start_date, end_date, load_data=True, create_model_data=True)
                predictions = {}
                y_test = None
                x_test_index = None

                for comparison in group:
                    model = comparison['model']
                    params = comparison.get('params', {})
                    features = stock_data.create_feature_list(comparison)
                    target = stock_data.create_target(comparison)
                    stock_data.split_data()
                    ML_data = stock_data.get_data2()
                    print(ML_data['x_test'])
                    stock_data.standardise_input(stock_data.features)
                    ML_data = stock_data.get_data2()
                    print(ML_data['x_test'])
                    model_handler = ModelHandler(ML_data, params)

                    try:
                        if model == 'LinearRegression':
                            model_handler.regression()
                        elif model == 'RNN':
                            model_handler.ML(model_handler.simpleRNN_)
                        elif model == 'LSTM':
                            model_handler.ML(model_handler.lstm_)

                        y_pred = model_handler.model.predict(ML_data['x_test'])
                        predictions[model] = y_pred
                        y_test = ML_data['y_test']
                        x_test_index = ML_data['x_test'].index
                    except Exception:
                        continue

                if predictions:
                    # Handle single comparison case
                    if len(predictions) == 1:
                        model_name, y_pred = next(iter(predictions.items()))
                        predictions = {model_name: y_pred}

                    prediction_chart_url = create_prediction_chart(x_test_index, y_test, predictions)
                    error = np.abs(y_test - list(predictions.values())[0].flatten())
                    error_chart_url = create_error_chart(x_test_index, error)

        # Get unique tickers for the dropdown
        unique_tickers = sorted(set(c['ticker'] for c in comparisons))

        return render_template(
            'comparison_results.html',
            results=results,
            unique_tickers=unique_tickers,
            selected_ticker=selected_ticker,
            prediction_chart_url=prediction_chart_url,
            error_chart_url=error_chart_url
        )

    @app.context_processor
    def inject_navigation():
        return dict(navigation=[
            {'name': 'Home', 'url': url_for('index')}
        ])
