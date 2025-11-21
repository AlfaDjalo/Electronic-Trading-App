"""API routes and request handling for the Flask backend.

This module declares the Flask endpoints used by the front-end. It
performs request validation, lightweight transformation (CSV/date
formatting) and delegates model runs to the service layer
(`services.model_service.process_models_request`).

Endpoints of interest:
    - /api/upload_data
    - /api/yahoo_data
    - /api/feature_sets
    - /api/run_models
"""

import pandas as pd
# import numpy as np
# import io
import os
from flask import request, jsonify
import json
import requests
from io import StringIO
# from flask_cors import CORS
# import traceback

# from stock_data import StockData
# from ml_data import MLData
# from model_handler import ModelHandler
from feature_set import FeatureSetManager, FEATURE_SETS_FILE
# from process_data import DataProcessor
from services.model_service import process_models_request
# from routes import get_tickers_by_category
# from flask import render_template, request, redirect, url_for, session, jsonify, send_file, send_from_directory  # Add this import for serving files

DEBUG = True

AVAILABLE_FUNCTIONS = [
    "raw_data",
    "create_lag",
    "create_average",
    "create_lagged_average",
    "create_rsi",
    "create_bollinger_bands",
]

# API Routes for Front-End
def setup_api_routes(app):
    """
    Set up all routes for the Flask application.

    Args:
        app (Flask): The Flask application instance.
    """

    @app.route("/api/upload_data", methods=["POST", "OPTIONS"])
    def upload_data():
        if request.method == "OPTIONS":
            # Preflight request handled automatically by flask-cors
            return jsonify({"status": "ok"}), 200

        if "fileName" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        fileName = request.files['fileName']
        
        try:
            raw_data = load_file(fileName)

            response = {
                "success": True,
                "timeSeriesData": raw_data,
            }

            return jsonify(response)

        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            print(f"Upload error: {error_msg}")
            return jsonify({'error': error_msg}), 500


    @app.route("/api/yahoo_data", methods=["POST", "OPTIONS"])
    def yahoo_data():
        if request.method == "OPTIONS":
            # Preflight request handled automatically by flask-cors
            return jsonify({"status": "ok"}), 200

        data = request.get_json()
        ticker = data.get("ticker")
        start_date = data.get("start_date")
        end_date = data.get("end_date")

        if not ticker:
            return jsonify({"success": False, "error": "Ticker is required"}), 400

        try:
            df = yf.download(tickers=ticker, start=start_date, end=end_date, auto_adjust=False)
            if df.empty:
                return jsonify({"success": False, "error": "No data found"}), 404

            df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
            df = df.reset_index()

            return jsonify({
                "success": True,
                "timeSeriesData": df.to_dict(orient="records"),
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500


    @app.route('/api/get_tickers/<category>', methods=['GET'])
    # @app.route('/api/get_tickers/<category>', methods=['GET', "OPTIONS"])
    # Maybe can move this into front-end ?
    def get_tickers(category):
        """
        Fetch tickers for the given category.

        Args:
            category (str): The selected category.

        Returns:
            Response: JSON response containing the list of tickers.
        """
            
        print(f"Getting tickers for {category}")
        tickers = get_tickers_by_category(category)
        return jsonify({'tickers': tickers})


    @app.route("/api/feature_sets", methods=["GET", "OPTIONS"])
    def get_feature_sets():
        if request.method == "OPTIONS":
            # Preflight request handled automatically by flask-cors
            return jsonify({"status": "ok"}), 200

        try:
            with open(FEATURE_SETS_FILE, "r") as f:
                feature_sets = json.load(f)
            return jsonify(feature_sets), 200

        except FileNotFoundError:
            return jsonify({"error": "feature_sets.json not found"}), 404

        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format"}), 500


    @app.route("/api/functions", methods=["GET"])
    def get_functions():
        return jsonify(AVAILABLE_FUNCTIONS)


    @app.route("/api/model_parameters")
    def get_model_config():
        file_path = os.path.join("config", "model_parameters.json")
        with open(file_path, "r") as f:
            data = f.read()
        return jsonify(eval(data))

    @app.route("/api/save_feature_set", methods=["POST"])
    def save_feature_set():
        try:
            data = request.json
            name = data.get("name")
            feature_set = data.get("feature_set")

            if not name or not feature_set:
                return jsonify({"error": "Missing feature set name or content"}), 400

            # Load existing file
            if os.path.exists(FEATURE_SETS_FILE):
                with open(FEATURE_SETS_FILE, "r") as f:
                    all_sets = json.load(f)
            else:
                all_sets = {}

            # Overwrite / update
            all_sets[name] = feature_set

            # Save back to file
            with open(FEATURE_SETS_FILE, "w") as f:
                json.dump(all_sets, f, indent=2)

            return jsonify({"status": "ok", "message": f"Feature set '{name}' saved."})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


    @app.route('/api/run_models', methods=['POST'])
    def run_models():
        """
        API endpoint to run ML models on time series data.
        """
        # Handle CORS preflight
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200
        
        # Parse request
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
        except Exception as e:
            return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400
        
        # Extract request components
        raw_data = data.get("rawData", [])
        model_list = data.get("modelList", [])
        hyperparameters = data.get("hyperparameters", {})
        
        if DEBUG:
            print("model_list")
            print(model_list)
            print("hyperparameters")
            print(hyperparameters)
            print("Top-level keys received:", list(data.keys()))
        
        try:
            # Process the request using the business logic function
            results = process_models_request(
                raw_data_list=raw_data,
                model_list=model_list,
                hyperparameters=hyperparameters,
                # feature_sets_file=FEATURE_SETS_FILE, # Maybe move this into model_service or even model_runner ?
                verbose=DEBUG
            )
            
            # Determine HTTP status based on results
            metadata = results.get("metadata", {})
            total_models = metadata.get("total_models", len(model_list))
            successful_models = metadata.get("successful_models", 0)
            
            if DEBUG:
                print("metadata")
                print(metadata)
                print("total_models")
                print(total_models)
                print("successful_models")
                print(successful_models)

            if successful_models == 0:
                return jsonify({
                    **results,
                    "error": "All models failed to process"
                }), 500
            elif successful_models < total_models:
                return jsonify({
                    **results,
                    "warning": f"Only {successful_models}/{total_models} models succeeded"
                }), 200
            else:
                return jsonify(results), 200
                
        except ValueError as e:
            # Client error (bad input)
            return jsonify({"error": str(e)}), 400
        
        except Exception as e:
            app.logger.exception("Unhandled exception in /api/run_models")
            return jsonify({
                "status": "error",
                "error": str(e),
                "results": []
            }), 500
   

def load_file(fileName):
    """
    Loads the requested file of asset price returns time series
    and formats the date column.

    Args:
        fileName: The name of the csv file to load. Must be in the correct format.

    Returns:

    """
    # Load CSV into pandas
    df = pd.read_csv(fileName)
    
    # Determine first column name and convert to datetime
    first_col = df.columns[0]

    # If column is named 'timestamp' or similar, treat as Unix epoch
    if "timestamp" in first_col.lower():
        # Determine if milliseconds or seconds
        sample_val = df[first_col].iloc[0]
        unit = "ms" if sample_val > 1e12 else "s"
        df[first_col] = pd.to_datetime(df[first_col], unit=unit, errors="coerce")
    else:
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce")

    # Rename first column to 'date' for consistency
    df.rename(columns={first_col: "date"}, inplace=True)

    # Move 'date' to be the first column
    cols = ["date"] + [c for c in df.columns if c != "date"]
    df = df[cols]

    # Convert datetime to ISO string
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    raw_data = df.to_dict(orient="records")

    return raw_data


def get_tickers_by_category(category: str):
    """
    Fetches tickers for a given category (e.g., 'australian', 'us').

    Tries to use lxml first, then html5lib. Returns empty list on failure.
    """
    url_map = {
        "australian": "https://en.wikipedia.org/wiki/S%26P/ASX_200",
        "us": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    }

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    url = url_map.get(category.lower())
    if not url:
        print(f"⚠️ Unknown category: {category}")
        return []

    print(f"Fetching tickers for {category} from {url}")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Try lxml first, then html5lib
        for flavor in ("lxml", "html5lib"):
            try:
                tables = pd.read_html(StringIO(response.text), flavor=flavor)
                break
            except Exception as e:
                print(f"⚠️ Failed to parse with {flavor}: {e}")
        else:
            print("❌ Could not parse tables with any parser.")
            return []

        # Extract tickers depending on table structure
        if category.lower() == "australian":
            df = tables[2]
            tickers = df[['Code', 'Company']].to_dict(orient='records')
        elif category.lower() == "us":
            df = tables[0]
            tickers = df[['Symbol', 'Security']].rename(
                columns={'Symbol': 'Code', 'Security': 'Company'}
            ).to_dict(orient='records')

        else:
            tickers = df[[]]

        print(f"✅ Found {len(tickers)} tickers for {category}")
        return tickers

    except Exception as e:
        print(f"❌ Error fetching tickers for {category}: {e}")
        return []

