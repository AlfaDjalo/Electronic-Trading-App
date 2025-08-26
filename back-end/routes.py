"""Manage routes for the Electronic Trading App."""

# Import necessary libraries
import base64
import json
import os  # Add this import for file handling
from flask import render_template, request, redirect, url_for, session, jsonify, send_file, send_from_directory  # Add this import for serving files
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from flask_caching import Cache
from datetime import date, datetime
from babel.numbers import format_decimal
import requests

# import app modules
from model_handler import ModelHandler
from stock_data import StockData, DEFAULT_START_DATE, DEFAULT_END_DATE, LOB_FILEPATH
from ml_data import MLData
from feature_set import FeatureSetManager
from charts import create_chart, TEMP_CHART_DIR
# from window_generator import WindowGenerator

DEBUG = False

def is_date(value):
    """Check if a value is a date or datetime."""
    return isinstance(value, (date, datetime))

def format_date(value, format="%Y-%m-%d"):
    """Custom Jinja2 filter to format dates."""
    if isinstance(value, (datetime, date)):
        return value.strftime(format)
    return value

def floatformat(value, precision=2):
    """Custom Jinja2 filter to format floats to a given precision."""
    try:
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return value

def intcomma(value):
    """Custom Jinja2 filter to add commas to large numbers."""
    try:
        return format_decimal(value, locale='en_US')
    except (ValueError, TypeError):
        return value

# Global constants for default values
# DEFAULT_LAG_PERIOD = 3
# DEFAULT_FORECAST_PERIOD = 1
FEATURE_SETS_FILE = "c:\\Users\\David\\Projects\\Electronic Trading App\\data\\feature_sets.json"
model_params_path = os.path.join(os.path.dirname(__file__), 'model_parameters.json')
 
def setup_routes(app):
    """
    Set up all routes for the Flask application.

    Args:
        app (Flask): The Flask application instance.
    """
    app.secret_key = 'your_secret_key'  # Add a secret key for session management
    comparison_counter = 1  # Initialize a counter for generating unique names

    
    # Your normal route logic here
    # Initialize cache
    cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
    cache.init_app(app)

    # Initialize chart data as a shared resource
    chart_data = {
        "input_data_chart": None,
        "prediction_chart": None,
        "error_chart": None
    }

    # Add the custom test to the Jinja2 environment
    app.jinja_env.tests['date'] = is_date
    app.jinja_env.filters['date'] = format_date  # Add the custom date filter
    app.jinja_env.filters['floatformat'] = floatformat  # Add the custom floatformat filter
    app.jinja_env.filters['intcomma'] = intcomma  # Add the custom intcomma filter

    feature_set_manager = FeatureSetManager(FEATURE_SETS_FILE)

    # @app.route("/", methods=["GET", "POST"])
    @app.route("/")
    def index():
        session.clear()
        category = session.get("category", "australian")
        tickers = get_tickers_by_category(category)
        return render_template(
            "index.html",
            category=category,
            tickers=tickers,
            session_ticker=session.get("ticker"),
            session_start_date=session.get("start_date", DEFAULT_START_DATE),
            session_end_date=session.get("end_date", DEFAULT_END_DATE),
            session=session, # Pass the session to the template if needed for LOB checkbox state
        )

    @app.route('/create_comparison', methods=['POST'])
    def create_comparison():
        """
        Create a new comparison and add it to the session.

        Returns:
            Response: Rendered comparison page with updated comparisons.
        """
        nonlocal comparison_counter  # Use the counter to generate unique names
        category = session.get('category', 'australian')  # Default to Australian stocks
        ticker = session.get('ticker')  # Use ticker from session
        if not ticker:
            return "Ticker is required", 400  # Return an error if ticker is missing
        if category == 'australian' and not ticker.endswith('.AX'):
            ticker += '.AX'  # Ensure Australian tickers have ".AX"
        start_date = session.get('start_date')  # Use start_date from session
        end_date = session.get('end_date')  # Use end_date from session
        model = request.form.get('model')
        has_lob_data = category == 'crypto'  # Flag LOB data for the crypto category

        # lag_period = request.form.get('lag_period', DEFAULT_LAG_PERIOD)
        # lag_period = int(lag_period) if lag_period.strip() else DEFAULT_LAG_PERIOD

        # forecast_period = request.form.get('forecast_period', DEFAULT_FORECAST_PERIOD)
        # forecast_period = int(forecast_period) if forecast_period.strip() else DEFAULT_FORECAST_PERIOD

        # Load full parameter metadata from model_parameters.json
        with open(model_params_path) as f:
            all_parameters = json.load(f)
        model_parameters = all_parameters.get(model, {})

        # Initialize the value field for each parameter with its default value
        for param, metadata in model_parameters.items():
            metadata['value'] = metadata.get('default')

        # Retrieve the list of comparisons from the session
        comparisons = session.get('comparisons', [])

        feature_set_name = request.form.get('feature_set_name')  # Retrieve selected feature set
        normalise = request.form.get('normalise') == 'on'

        # Create a new comparison dictionary
        comparison = {
            'name': f"Comparison_{comparison_counter}",  # Generate unique name
            'model': model,
            # 'lag_period': lag_period,  # Add lag_period
            # 'forecast_period': forecast_period,  # Add forecast_period
            'params': model_parameters,  # Include full parameter metadata with initialized values
            # 'use_log_returns': False,  # Default to off
            # 'use_lob_data': False,  # Include LOB data flag
            'feature_set_name': feature_set_name,  # Add selected feature set
            'normalise': normalise  # Add normalization status
        }
        comparison_counter += 1  # Increment the counter

        comparisons.append(comparison)  # Add the dictionary directly
        session['comparisons'] = comparisons  # Save the updated list back to the session

        # ...existing code...

        # Reload the comparison page with updated comparisons
        with open(model_params_path) as f:
            models = list(json.load(f).keys())  # Load model names from JSON
        # feature_sets = session.get('feature_sets', {})  # Load feature sets from session
        data_type = session.get('data_type', 'daily')  # Retrieve data type from session
        filtered_feature_set_names = feature_set_manager.get_feature_sets_by_data_type(data_type)
        # filtered_feature_sets = {
        #     name: fs for name, fs in feature_sets.items()
        #     if fs.get("data_type") == data_type
        # }  # Filter feature sets by data type

        return render_template(
            "comparison_page.html",
            comparisons=comparisons,
            models=models,
            feature_set_names=filtered_feature_set_names,  # Pass filtered feature sets
            data_type=data_type  # Pass data type to template
        )

    @app.route('/set_parameters/<int:index>', methods=['GET', 'POST'])
    def set_parameters(index):
        """
        Set or update parameters for a specific comparison.

        Args:
            index (int): Index of the comparison to update.

        Returns:
            Response: Rendered HTML template or redirect to the home page.
        """
        comparisons = session.get('comparisons', [])
        if index >= len(comparisons):
            return "Comparison not found", 404

        if request.method == 'POST':
            parameters = request.form.to_dict()
            model_metadata = comparisons[index]['params']
            # ...existing code...
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
                # ...existing code...
                # Save validated value back to the metadata dictionary
                metadata['value'] = value

            use_log_returns = request.form.get('use_log_returns', 'off') == 'on'
            comparisons[index]['use_log_returns'] = use_log_returns

            # Save updated parameters to the comparison and session
            comparisons[index]['params'] = model_metadata
            session['comparisons'] = comparisons

            return redirect(url_for('comparison_page'))  # Redirect to comparison_page instead of index

        model = comparisons[index]['model']
        model_parameters = comparisons[index]['params']

        # ...existing code...
        # Ensure metadata['values'] is a list
        for param, metadata in model_parameters.items():
            if metadata.get('type') == 'category' and callable(metadata.get('values')):
                metadata['values'] = list(metadata['values']())

        # ...existing code...
        # Retrieve ticker, start_date, and end_date from the session
        ticker = session.get('ticker')  # Retrieve ticker from session
        start_date = session.get('start_date')  # Retrieve start_date from session
        end_date = session.get('end_date')  # Retrieve end_date from session

        return render_template(
            'set_parameters.html',
            model=model,
            model_parameters=model_parameters,
            comparison=comparisons[index],
            use_log_returns=comparisons[index].get('use_log_returns', False),
            normalise=comparisons[index].get('params', {}).get('normalise', {}).get('value', False),  # Pass normalise
            ticker=ticker,  # Pass ticker from session
            start_date=start_date,  # Pass start_date from session
            end_date=end_date  # Pass end_date from session
        )

    @app.route('/edit_comparison/<int:index>', methods=['GET', 'POST'])
    def edit_comparison(index):
        """
        Edit the details of an existing comparison.

        Args:
            index (int): Index of the comparison to edit.

        Returns:
            Response: Rendered HTML template or redirect to the comparison page.
        """
        comparisons = session.get('comparisons', [])
        if index >= len(comparisons):
            return "Comparison not found", 404

        if request.method == 'POST':
            if 'cancel' in request.form:
                return redirect(url_for('comparison_page'))  # Redirect to comparison_page on cancel

            new_name = request.form.get('name')
            existing_names = {c['name'] for i, c in enumerate(comparisons) if i != index}
            
            data_type = session.get('data_type', 'daily')
            filtered_feature_set_names = feature_set_manager.get_feature_sets_by_data_type(data_type)

            if new_name in existing_names:
                error_message = f"A comparison with the name '{new_name}' already exists."
                return render_template(
                    'edit_comparison.html',
                    comparison=comparisons[index],
                    feature_set_names=filtered_feature_set_names,
                    data_type=data_type,
                    error_message=error_message
                )

            comparisons[index]['name'] = new_name  # Update the name
            comparisons[index]['model'] = request.form.get('model')
            comparisons[index]['feature_set_name'] = request.form.get('feature_set_name')
            comparisons[index]['normalise'] = request.form.get('normalise') == 'on'
            session['comparisons'] = comparisons
            return redirect(url_for('comparison_page'))  # Redirect to comparison_page after saving

        # Load feature sets and filter by data type
        # feature_sets = session.get('feature_sets', {})
        data_type = session.get('data_type', 'daily')
        filtered_feature_set_names = feature_set_manager.get_feature_sets_by_data_type(data_type)
        # filtered_feature_sets = {
        #     name: fs for name, fs in feature_sets.items()
        #     if fs.get("data_type") == data_type
        # }

        return render_template(
            'edit_comparison.html',
            comparison=comparisons[index],
            feature_set_names=filtered_feature_set_names,
            data_type=data_type,
            error_message=None
        )

    @app.route('/delete_comparison/<int:index>', methods=['POST', 'GET'])
    def delete_comparison(index):
        """
        Delete a comparison from the session.

        Args:
            index (int): Index of the comparison to delete.

        Returns:
            Response: Redirect to the comparison page.
        """
        comparisons = session.get('comparisons', [])
        if index >= len(comparisons):
            return "Comparison not found", 404

        comparisons.pop(index)
        session['comparisons'] = comparisons
        return redirect(url_for('comparison_page'))  # Redirect to comparison_page

    @app.route('/clear_comparisons', methods=['POST'])
    def clear_comparisons():
        """
        Clear all comparisons from the session.

        Returns:
            Response: Redirect to the home page.
        """
        session['comparisons'] = []
        return redirect(url_for('index'))

    @app.route('/run_comparisons', methods=['POST'])
    def run_comparisons():
        """
        Run all comparisons and generate results.

        Returns:
            Response: Redirect to the comparison_results page.
        """
        nonlocal chart_data  # Access chart_data from the enclosing scope
        comparisons = session.get('comparisons', [])
        data_type = session.get('data_type')
        ticker = session.get('ticker')  # Use ticker directly from the session
        # use_lob_data = session.get('use_lob_data', False)  # Retrieve LOB data flag from session
        # global LOB_FILEPATH
        # LOB_FILEPATH = "path/to/lob_data.csv" if session.get('use_lob_data', False) else None  # Set global LOB_FILEPATH

        results = []
        predictions = {}
        errors = {}
        y_test = None
        x_test_index = None

        for comparison in comparisons:
            try:
                # Load stock data
                stock_data = StockData(
                    data_type,
                    ticker, 
                    session.get('start_date'), 
                    session.get('end_date'), 
                    load_data=True,
                    verbose=DEBUG
                )

                # Load feature set
                feature_set_name=comparison.get('feature_set_name', None)

                # Create machine learning data
                ml_data = MLData(
                    raw_data=stock_data.get_data(),
                    train_percentage=0.8,
                    feature_set = feature_set_manager.get_feature_set(feature_set_name),
                    normalise=comparison.get('normalise', False),
                    verbose=DEBUG
                )

                # Create model handler
                model_handler = ModelHandler(ml_data.get_data(), params=comparison.get('params', {}), window_generator=ml_data.get_window(), target=ml_data.get_target(), verbose=DEBUG)
                model = comparison['model']

                model_handler.run_keras_model(model)

                # if model == "Baseline":
                #     # model_handler.baseline(ml_data.get_target())
                #     model_handler.baseline(ml_data.get_window(), ml_data.get_target())
                # elif model == 'LinearRegression':
                #     model_handler.keras_regression(ml_data.get_window(), ml_data.get_target())
                #     # model_handler.regression()
                # elif model == 'RNN':
                #     model_handler.ML(model_handler.simpleRNN_)
                # elif model == 'LSTM':
                #     model_handler.keras_LSTM(ml_data.get_window(), ml_data.get_target())
                #     # model_handler.ML(model_handler.lstm_)
                # elif model == 'GRU':
                #     model_handler.ML(model_handler.gru_)
                # # elif model == 'AlphaRNN':
                # #     model_handler.ML(model_handler.alpharnn_)
                # # elif model == 'AlphatRNN':
                # #     model_handler.ML(model_handler.alphatrnn_)
                # elif model == 'CNN':
                #     model_handler.keras_CNN(ml_data.get_window(), ml_data.get_target())
                #     # model_handler.train_lob_cnn()
                #     # return redirect(url_for('comparison_results'))
                # elif model == 'MLP':
                #     model_handler.keras_MLP(ml_data.get_window(), ml_data.get_target())

                print(f"Predicting for {model}")

                # print_session_size(session)

                # print("Current session keys:", list(session.keys()))
                # print("Session size:", len(str(session)))

                # json_size = len(json.dumps(dict(session), default=str).encode('utf-8'))
                # print(f"JSON serialized size: {json_size} bytes")

                # estimated_cookie_size = int(json_size * 1.4)  # Account for encoding overhead
                # print(f"Estimated cookie size: {estimated_cookie_size} bytes")

                if model == 'CNN_old':
                    x_test = model_handler.prepare_cnn_input(ml_data.get_data()['x_test'])
                    y_pred = model_handler.model.predict(x_test)
                    y_test = ml_data.get_data()['y_test'].values
                elif model in ["baseline", "keras_regression", "keras_LSTM", "keras_CNN", "keras_MLP"]:
                    print(f"Predicting for {model} model.")
                    # performance = model_handler.model.evaluate(ml_data.get_window().test, return_dict=True)
                    test_inputs = np.concatenate([inputs.numpy() for inputs, labels in ml_data.get_window().test])
                    y_pred = model_handler.model.predict(test_inputs)
  
                    # y_pred = model_handler.model.predict(inputs=ml_data.get_window().test)
                    print("y_pred shape:", y_pred.shape)
                    # y_test = ml_data.get_window().test.map(lambda inputs, labels: labels)
                    y_test = np.concatenate([labels.numpy() for labels in ml_data.get_window().test.map(lambda inputs, labels: labels)])
                    print("y_test shape:", y_test.shape)
                    # y_pred = model_handler.model.predict(ml_data.get_data()['test_df'])

                    # Reshape to 2D by removing last dimension
                    y_pred = y_pred.reshape(y_pred.shape[0], -1)  # (1198, 3)
                    y_test = y_test.reshape(y_test.shape[0], -1)  # (1198, 1)

                    print("y_pred shape:", y_pred.shape)
                    print("y_test shape:", y_test.shape)

                else:
                    y_pred = model_handler.model.predict(ml_data.get_data()['x_test'])
                    print("y_pred shape:", y_pred.shape)
                    y_test = ml_data.get_data()['y_test'].values
                    print("y_test shape:", y_test.shape)

                # Reverse normalization for both predictions and actual values
                if ml_data.get_normalise():
                    # Retrieve normalisation parameters for the target
                    target_params = ml_data.get_normalisation_params(ml_data.get_target())
                    y_pred = (y_pred * target_params['std']) + target_params['mean']
                    y_test = (ml_data.get_data()['y_test'].values * target_params['std']) + target_params['mean']
                # else:
                #     y_test = ml_data.get_data()['y_test'].values

                if x_test_index is None:
                    x_test_index = ml_data.get_data()['x_test'].index

                predictions[comparison['name']] = y_pred
                errors[comparison['name']] = y_test - y_pred


                results.append({
                    'name': comparison['name'],
                    'model': model,
                    'stats': model_handler.get_stats(y_test, y_pred),
                    'error': 'N/A'
                })
            except ValueError as e:
                print(f"❌ DATA ERROR: {str(e)}")
                results.append({
                    'name': comparison['name'],
                    'model': comparison['model'],
                    'stats': 'N/A',
                    'error': f"Data Error: {str(e)}"
                })
            except Exception as e:
                print(f"❌ CHART ERROR: {str(e)}")
                results.append({
                    'name': comparison['name'],
                    'model': comparison['model'],
                    'stats': 'N/A',
                    'error': str(e)
                })

        if x_test_index is not None and y_test is not None and predictions:
            prediction_chart_path = create_chart(x_test_index, {"Actual": y_test, **predictions}, chart_type='prediction')
            error_chart_path = create_chart(x_test_index, errors, chart_type='error')
            session['chart_paths'] = {
                'prediction_chart': prediction_chart_path,
                'error_chart': error_chart_path
            }

        # print(results)
        # results_size = len(json.dumps(results, default=str).encode('utf-8'))
        # print(f"🔍 Results data size: {results_size} bytes")
        
        # Store results in the session for rendering on the comparison_results page
        session['comparison_results'] = {
            'results': results,
            'selected_ticker': ticker
        }

        return redirect(url_for('comparison_results'))

    @app.route('/comparison_results', methods=['GET'])
    def comparison_results():
        """
        Display the results of the comparisons.

        Returns:
            Response: Rendered HTML template with comparison results.
        """
        comparison_results = session.get('comparison_results', {})
        chart_paths = session.get('chart_paths', {})
        return render_template(
            'comparison_results.html',
            results=comparison_results.get('results', []),
            selected_ticker=comparison_results.get('selected_ticker'),
            prediction_chart_url=url_for('serve_chart', filename=os.path.basename(chart_paths.get('prediction_chart', ''))),
            error_chart_url=url_for('serve_chart', filename=os.path.basename(chart_paths.get('error_chart', '')))
        )

    @app.route('/view_charts', methods=['GET','POST'])
    def view_charts():
        """
        Display the results of the comparisons.

        Returns:
            Response: Rendered HTML template with comparison results.
        """
        # comparison_results = session.get('comparison_results', {})

        ticker = session.get('ticker')  # Use ticker from session
        start_date = session.get('start_date')  # Use start_date from session
        end_date = session.get('end_date')  # Use end_date from session
        data_type = session.get('data_type')

        if request.method == 'POST':
            stock_data = StockData(
                data_type,
                ticker, 
                start_date, 
                end_date, 
                load_data=True
            )

            # Select the appropriate data type
            raw_data = stock_data.get_data()
            features = stock_data.get_available_fields()        
        
            feature = request.form.get('feature')

            time_series = raw_data.index
            feature_series = {feature: raw_data[feature].values}

            feature_chart_path = create_chart(time_series, feature_series, chart_type='feature')
            if 'chart_paths' not in session:
                session['chart_paths'] = {
                    'feature_chart': feature_chart_path
                    }
            else:
                # print(session['chart_paths'])
                session['chart_paths']['feature_chart'] = feature_chart_path
    
            # {
            #     'prediction_chart': prediction_chart_path,
            #     'error_chart': error_chart_path
            #     'feature_chart': feature_chart_path
            # }
        else: 
            available_fields_data = load_available_fields()
            features = available_fields_data.get(data_type, {}).get("fields", [])
            feature = features[0]

        chart_paths = session.get('chart_paths', {})
        return render_template(
            'view_charts.html',
            selected_feature=feature,
            features=features,
            feature_chart_url=url_for('serve_chart', filename=os.path.basename(chart_paths.get('feature_chart', '')))
        )

    @app.route('/chart/<filename>')
    def serve_chart(filename):
        """
        Serve a chart image from the temporary directory.

        Args:
            filename (str): Name of the chart file.

        Returns:
            Response: The requested chart image.
        """
        return send_from_directory(TEMP_CHART_DIR, filename)


    @app.route('/test_stock_data/<int:index>', methods=['GET', 'POST'])
    def test_stock_data(index):
        """
        Test the functionality of the StockData and MLData classes and display data samples.

        Args:
            index (int): Index of the comparison to test.

        Returns:
            Response: Rendered HTML template with data samples.
        """
        ticker = session.get('ticker')  # Use ticker from session
        start_date = session.get('start_date')  # Use start_date from session
        end_date = session.get('end_date')  # Use end_date from session
        data_type = session.get('data_type')
        # stock_data = session.get('stock_data')
        
        if index==9999:
            # print("Called from top button.")
            return render_template('test_stock_data.html', data_preview=None, error_message='Not built yet.')
        # else:
            # print("Called from comparison.")

        comparisons = session.get('comparisons', [])
        if index >= len(comparisons):
            return "Comparison not found", 404

        comparison = comparisons[index]
        # lag_period = int(comparison['params'].get('num_days_lag', {}).get('value', DEFAULT_LAG_PERIOD))
        # forecast_period = int(comparison['params'].get('forward_projection_days', {}).get('value', DEFAULT_FORECAST_PERIOD))
        # log_returns = comparison.get('use_log_returns', False)
        normalise = comparison.get('normalise', {})
        # use_lob_data = session.get('use_lob_data', False)  # Retrieve LOB data flag from session
        # Check if processed data is cached
        cache_key = f"processed_data_{index}"
        processed_data = cache.get(cache_key)

        if not processed_data:
            try:
                # Create StockData object
                stock_data = StockData(
                    data_type,
                    ticker, 
                    start_date, 
                    end_date, 
                    load_data=True
                )

                # Select the appropriate data type
                raw_data = stock_data.get_data()
                feature_set_name = comparison.get('feature_set_name')
                feature_set = feature_set_manager.get_feature_set(feature_set_name)

                # Create MLData object
                ml_data = MLData(
                    raw_data=raw_data,
                    # lag_period=lag_period,
                    # forecast_period=forecast_period,
                    # log_returns=log_returns,
                    normalise=normalise,
                    # data_type='intraday' if use_lob_data else 'daily',
                    feature_set=feature_set  # Pass feature_set
                )

                # Convert data to JSON-serializable format
                processed_data = {
                    'raw': raw_data.reset_index().to_dict(orient='records'),
                    'x_train': ml_data.get_data()['x_train'].reset_index().to_dict(orient='records') if ml_data.get_data()['x_train'] is not None else [],
                    'x_test': ml_data.get_data()['x_test'].reset_index().to_dict(orient='records') if ml_data.get_data()['x_test'] is not None else [],
                    'y_train': ml_data.get_data()['y_train'].reset_index().to_dict(orient='records') if ml_data.get_data()['y_train'] is not None else [],
                    'y_test': ml_data.get_data()['y_test'].reset_index().to_dict(orient='records') if ml_data.get_data()['y_test'] is not None else []
                }

                # Cache the processed data
                cache.set(cache_key, processed_data, timeout=300)  # Cache for 5 minutes
            except Exception as e:
                return render_template('test_stock_data.html', data_preview=None, error_message=str(e))

        data_preview = None
        error_message = None
        datasets = ['raw', 'x_train', 'x_test', 'y_train', 'y_test']  # Available data types
        selected_dataset = request.args.get('dataset', datasets[0])  # Default to the first item

        try:
            # Retrieve processed data from the cache
            data = pd.DataFrame(processed_data[selected_dataset])

            # Get the first and last five rows
            data_preview = {
                'head': data.head(5),
                'tail': data.tail(5)
            }
        except Exception as e:
            error_message = str(e)

        return render_template(
            'test_stock_data.html',
            data_preview=data_preview,
            error_message=error_message,
            datasets=datasets,
            selected_dataset=selected_dataset  # Pass the selected item
        )


    AVAILABLE_FIELDS_FILE = "c:\\Users\\David\\Projects\\Electronic Trading App\\data\\available_fields.json"

    def load_available_fields():
        """
        Load available fields from the available_fields.json file.

        Returns:
            dict: A dictionary of available fields grouped by data type.
        """
        try:
            with open(AVAILABLE_FIELDS_FILE, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return {}

    @app.route('/manage_feature_sets', methods=['GET', 'POST'])
    def manage_feature_sets():
        data_type = session.get('data_type')
        feature_set_names = feature_set_manager.get_feature_sets_by_data_type(data_type)

        selected_feature_set = request.form.get('feature_set') or session.get('selected_feature_set') or next(iter(feature_set_names), None)

        if selected_feature_set == 'add_new':
            selected_feature_set = None
            features = []
            target = None
        else:
            session['selected_feature_set'] = selected_feature_set
            fs = feature_set_manager.get_feature_set(selected_feature_set) if selected_feature_set else {}
            features = fs.get("features", [])
            target = fs.get("target", None)

        # Store temporary changes in the session
        if 'temp_feature_set' not in session:
            session['temp_feature_set'] = {}

        temp_feature_set = session['temp_feature_set'].get(selected_feature_set, {"features": features, "target": target})

        for feature in temp_feature_set["features"]:
            if 'function_parameters' not in feature:
                feature['function_parameters'] = None

        available_fields_data = load_available_fields()
        available_data_fields = available_fields_data.get(data_type, {}).get("fields", [])

        if request.method == 'POST':
            action = request.form.get('action')
            if action:
                if action == 'create_target':
                    temp_feature_set["target"] = {
                        "name": "target",
                        "input_data_fields": [],
                        "function": "raw_data",
                        "function_parameters": None
                    }

                elif action == 'delete_target':
                    temp_feature_set["target"] = None

                elif action == 'add_feature':
                    input_data_fields = request.form.getlist('data_fields')  # Get selected input fields
                    function = request.form.get('function')  # Get selected function
                    try:
                        function_parameters = json.loads(request.form.get('function_parameters', '{}'))  # Parse JSON parameters
                    except json.JSONDecodeError:
                        function_parameters = {}

                    # Generate the feature name based on the function
                    if function == 'raw_data':
                        if len(input_data_fields) != 1:
                            return "raw_data function requires exactly one input field.", 400
                        feature_name = input_data_fields[0]
                    elif function == 'create_lag':
                        if len(input_data_fields) != 1:
                            return "create_lag function requires exactly one input field.", 400
                        num_lags = function_parameters.get("num_lags", 1)
                        feature_name = f"{input_data_fields[0]}_lagged_{num_lags}"
                    elif function == 'create_average':
                        if len(input_data_fields) < 2:
                            return "create_average function requires at least two input fields.", 400
                        feature_name = f"avg_{'_'.join(input_data_fields)}"
                    elif function == 'create_rsi':
                        if len(input_data_fields) != 1:
                            return "create_rsi function requires exactly one input field.", 400
                        window = function_parameters.get("window", 20)
                        feature_name = f"{input_data_fields[0]}_rsi"
                    else:
                        return f"Unsupported function: {function}", 400

                    # Create the new feature dictionary
                    new_feature = {
                        "name": feature_name,
                        "input_data_fields": input_data_fields,
                        "function": function,
                        "function_parameters": function_parameters
                    }

                    # Add the new feature to the temporary feature set
                    temp_feature_set["features"].append(new_feature)

                elif action.startswith("delete_feature_"):
                    try:
                        index = int(action.split("_")[-1])
                        if 0 <= index < len(temp_feature_set["features"]):
                            del temp_feature_set["features"][index]
                    except Exception as e:
                        print("Error deleting feature:", e)

                elif action == 'save':
                    # Save the temporary feature set to the JSON file
                    if selected_feature_set:
                        feature_set_manager.update_feature_set(selected_feature_set, temp_feature_set)
                    session['temp_feature_set'].pop(selected_feature_set, None)  # Clear temporary changes after saving
                    return redirect(url_for('manage_feature_sets'))

                elif action == 'delete_set':
                    if selected_feature_set:
                        feature_set_manager.delete_feature_set(selected_feature_set)
                        session.pop('selected_feature_set', None)
                    session['temp_feature_set'].pop(selected_feature_set, None)  # Clear temporary changes
                    return redirect(url_for('manage_feature_sets'))

                elif action == 'cancel':
                    # Revert temporary changes by reloading the feature set from the JSON file
                    session['temp_feature_set'].pop(selected_feature_set, None)
                    return redirect(url_for('manage_feature_sets'))

                elif action == 'exit':
                    # Revert temporary changes by reloading the feature set from the JSON file
                    session['temp_feature_set'].pop(selected_feature_set, None)
                    return redirect(url_for('manage_feature_sets'))

            # Update the session with the temporary changes
            session['temp_feature_set'][selected_feature_set] = temp_feature_set

        return render_template(
            'manage_feature_sets.html',
            feature_set_names=feature_set_names,
            selected_feature_set=selected_feature_set,
            features=temp_feature_set["features"],
            target=temp_feature_set["target"],
            available_data_fields=available_data_fields
        )

    @app.route('/add_feature_set', methods=['GET', 'POST'])
    def add_feature_set():
        # print("In add_feature_set")
        if request.method == 'POST':
            feature_set_name = request.form.get('feature_set_name')

            request.form.getlist('features')
            feature_set_data = {feature: {} for feature in features}
            feature_set_manager.add_feature_set(feature_set_name, feature_set_data)
            return redirect(url_for('manage_feature_sets'))
        
        return render_template('add_feature_set.html', fields=fields)

    @app.route('/edit_feature_set/<feature_set_name>', methods=['GET', 'POST'])
    def edit_feature_set(feature_set_name):
        feature_set = feature_set_manager.get_feature_set(feature_set_name)
        if not feature_set:
            return "Feature set not found", 404

        if request.method == 'POST':
            updated_features = request.form.getlist('features')
            feature_set_data = {feature: {} for feature in updated_features}
            feature_set_manager.update_feature_set(feature_set_name, feature_set_data)
            return redirect(url_for('manage_feature_sets'))

        fields = ["Open", "High", "Low", "Close", "Volume"]  # Example fields
        current_features = list(feature_set.keys())
        return render_template('edit_feature_set.html', feature_set_name=feature_set_name, fields=fields, features=current_features)

    @app.route('/save_feature_sets', methods=['POST'])
    def save_feature_sets():
        """
        Save all feature sets to a JSON file.
        """
        feature_sets = session.get('feature_sets', {})
        with open('data/feature_sets.json', 'w') as f:
            json.dump(feature_sets, f, indent=4)
        return redirect(url_for('comparison_page'))

    @app.context_processor
    def inject_navigation():
        """
        Inject navigation links into all templates.

        Returns:
            dict: Dictionary of navigation links.
        """
        return dict(navigation=[
            {'name': 'Home', 'url': url_for('index')}
        ])

    @app.route("/comparison_page")
    def comparison_page():
        """
        Display the page that manages the creation of comparisons.

        Returns:
            Response: Rendered comparison page with updated comparisons.
        """
        comparisons = session.get("comparisons", [])
        with open('model_parameters.json') as f:
            models = list(json.load(f).keys())  # Load model names from JSON
        
        data_type = session.get('data_type', 'daily')  # Retrieve data type from session
        feature_set_names = feature_set_manager.get_feature_sets_by_data_type(data_type)
        return render_template(
            "comparison_page.html",
            comparisons=comparisons,
            models=models,
            feature_set_names=feature_set_names,
            data_type=data_type,
        )

    @app.route('/set_ticker_and_dates', methods=['POST'])
    def set_ticker_and_dates():
        """
        Set the ticker and date range in the session after validation.

        Returns:
            Response: Redirect to the comparison page if valid, or back to the index page with an error message.
        """
        session["category"] = request.form.get("category", "australian")
        ticker = request.form.get("ticker")
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        use_lob_data = request.form.get("use_lob_data", "off") == "on"  # Handle LOB data checkbox

        # ...existing code...
        # Validate ticker
        if not ticker:
            return "Ticker is required", 400
        if session["category"] == 'australian' and not ticker.endswith('.AX'):
            ticker += '.AX'

        # ...existing code...
        # Validate dates
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
            if start_date >= end_date:
                return "Start date must be before end date", 400
        except ValueError:
            return "Invalid date format. Use YYYY-MM-DD.", 400

        # ...existing code...
        # Save validated data to session
        session["ticker"] = ticker
        session["start_date"] = start_date.strftime("%Y-%m-%d")
        session["end_date"] = end_date.strftime("%Y-%m-%d")
        # session["use_lob_data"] = use_lob_data  # Save LOB data flag to session
        session["data_type"] = "intraday" if use_lob_data else "daily"  # Set data_type based on use_lob_data
        session["comparisons"] = []  # Clear comparisons when changing ticker or dates

        if session["category"] == 'test':
            session["data_type"] = 'test'
        print(session["data_type"])
        return redirect(url_for("comparison_page"))

    @app.route('/get_tickers/<category>', methods=['GET'])
    def get_tickers(category):
        """
        Fetch tickers for the given category.

        Args:
            category (str): The selected category.

        Returns:
            Response: JSON response containing the list of tickers.
        """
        tickers = get_tickers_by_category(category)
        return jsonify({'tickers': tickers})

    def get_tickers_by_category(category):
        # Helper function to fetch tickers based on category
        tickers = []

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        if (category == 'australian'):
            url = "https://en.wikipedia.org/wiki/S%26P/ASX_200"
            response = requests.get(url, headers=headers)
            asx200 = pd.read_html(response.text)[2]
            # asx200 = pd.read_html('https://en.wikipedia.org/wiki/S%26P/ASX_200')[2]
            tickers = asx200[['Code', 'Company']].to_dict(orient="records")
        elif (category == 'us'):
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(url, headers=headers)
            sp500 = pd.read_html(response.text)[0]
            # sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
            tickers = sp500[['Symbol', 'Security']].rename(columns={'Symbol': 'Code', 'Security': 'Company'}).to_dict(orient="records")
        elif (category == 'fx'):
            tickers = [{'Code': 'AUDUSD=X', 'Company': 'AUDUSD'}]
        elif (category == 'crypto'):
            tickers = [{'Code': 'BTC-USD', 'Company': 'Bitcoin'}]
        elif (category == 'test'):
            tickers = [{'Code': 'flat', 'Company': 'Flat Co.'}, {'Code': 'ramp', 'Company': 'Ramp Co.'}, {'Code': 'wave', 'Company': 'Wave Co.'}]
        return tickers


AVAILABLE_FIELDS_FILE = "c:\\Users\\David\\Projects\\Electronic Trading App\\data\\available_fields.json"  # Add this line

def print_session_size(session):
    print("Current session keys:", list(session.keys()))
    print("Session size:", len(str(session)))

    json_size = len(json.dumps(dict(session), default=str).encode('utf-8'))
    print(f"JSON serialized size: {json_size} bytes")

    estimated_cookie_size = int(json_size * 1.4)  # Account for encoding overhead
    print(f"Estimated cookie size: {estimated_cookie_size} bytes")