import pandas as pd
import numpy as np
import io
import os
from flask import request, jsonify
import json
import requests
from io import StringIO
from flask_cors import CORS
import traceback

from stock_data import StockData
from ml_data import MLData
from model_handler import ModelHandler
from feature_set import FeatureSetManager
from process_data import DataProcessor
from services.model_service import process_models_request
# from routes import get_tickers_by_category
# from flask import render_template, request, redirect, url_for, session, jsonify, send_file, send_from_directory  # Add this import for serving files

# FEATURE_SETS_FILE = "c:\\Users\\David\\Projects\\electronic_trading_app\\back-end\\data\\feature_sets.json"
DEBUG = False
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# DATA_PATH = os.path.join(BASE_DIR, "data", "feature_sets.json")
# FEATURE_SETS_FILE = os.path.join(os.path.dirname(__file__), 'data\\feature_sets.json')
FEATURE_SETS_FILE = os.path.join(BASE_DIR, "back-end", "data", "feature_sets.json")


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
            # print(traceback.format_exc())
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

        print(data)

        if not ticker:
            return jsonify({"success": False, "error": "Ticker is required"}), 400

        try:
            df = yf.download(tickers=ticker, start=start_date, end=end_date, auto_adjust=False)
            print(df)
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
    def get_tickers(category):
        """
        Fetch tickers for the given category.

        Args:
            category (str): The selected category.

        Returns:
            Response: JSON response containing the list of tickers.
        """
        # if request.method == "OPTIONS":
        #     # Preflight request handled automatically by flask-cors
        #     return jsonify({"status": "ok"}), 200
            
        print(f"Getting tickers for {category}")
        tickers = get_tickers_by_category(category)
        return jsonify({'tickers': tickers})

    @app.route("/api/feature_sets", methods=["GET", "OPTIONS"])
    def get_feature_sets():
        if request.method == "OPTIONS":
            # Preflight request handled automatically by flask-cors
            return jsonify({"status": "ok"}), 200

        print(FEATURE_SETS_FILE)
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
        return jsonify(eval(data))  # or json.load(f)

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


# @app.route("/api/process_data", methods=["POST"])
# def process_data():
#     # 1. Parse request
#     payload = request.get_json()
#     feature_set_name = payload.get("feature_set_name")
#     feature_set_config = load_feature_set(feature_set_name)  # load JSON definition
#     normalise = payload.get("normalise", False)
#     train_ratio = payload.get("train_ratio", 0.8)
#     val_ratio = payload.get("val_ratio", 0.1)
#     window_params = payload.get("window_params", None)

#     # 2. Load raw data
#     raw_df = DataManager("uploads/latest.csv").load()

#     # 3. Apply features
#     engineered_df = FeatureEngineer(feature_set_config).apply(raw_df)

#     # 4. Split & preprocess
#     preproc = Preprocessor(normalise=normalise,
#                            train_ratio=train_ratio,
#                            val_ratio=val_ratio)
#     dataset = preproc.split(engineered_df,
#                             features=feature_set_config["features"],
#                             target=feature_set_config["target"])
#     if normalise:
#         preproc.fit_normalisation(dataset)
#         dataset = preproc.apply_normalisation(dataset)

#     # 5. Optional: windowing
#     if window_params:
#         wg = WindowGenerator(**window_params)
#         dataset = wg.make_windows(dataset)

#     # 6. Return structured response
#     return jsonify({
#         "metadata": {
#             "feature_set": feature_set_name,
#             "features": [f["name"] for f in feature_set_config["features"]],
#             "target": feature_set_config["target"]["name"],
#             "normalised": normalise,
#             "train_ratio": train_ratio,
#             "val_ratio": val_ratio,
#         },
#         "splits": {
#             "train": dataset.x_train.to_dict(orient="records"),
#             "val": dataset.x_val.to_dict(orient="records"),
#             "test": dataset.x_test.to_dict(orient="records"),
#         }
#     })

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
        
        # print("raw_data")
        # print(raw_data)
        if DEBUG:
            print("model_list")
            print(model_list)
            print("hyperparameters")
            print(hyperparameters)

        # if request.args.get('debug'):
            print("Top-level keys received:", list(data.keys()))
        
        try:
            # Process the request using the business logic function
            results = process_models_request(
                raw_data_list=raw_data,
                model_list=model_list,
                hyperparameters=hyperparameters,
                feature_sets_file=FEATURE_SETS_FILE,
                verbose=DEBUG
                # verbose=bool(request.args.get('verbose'))
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
        # except Exception as e:
        # # Unexpected runtime error → still return JSON, not generic 500
        #     traceback.print_exc()  # logs full error to test output
        #     return jsonify({"error": str(e)}), 500
        
        # except Exception as e:
        #     # Server error
        #     return jsonify({
        #         "error": "Internal server error",
        #         "details": str(e) if app.debug else "Contact support"
        #     }), 500




    # def run_models():
    #     """
    #     Receive model configs and time series data, run ML models,
    #     and return predictions & stats as JSON.
    #     """

    #     if request.method == "OPTIONS":
    #         # Preflight request handled automatically by flask-cors
    #         return jsonify({"status": "ok"}), 200
        
    #     data = request.get_json()
    #     print("Top-level keys received:", list(data.keys()))

    #     raw_data = data.get("rawData", [])
    #     model_list = data.get("modelList", [])
    #     hyperparameters = data.get("hyperparameters")

    #     # Validation
    #     if not model_list:
    #         return jsonify({"error": "No models provided"}), 400
    #     if not raw_data:
    #         return jsonify({"error": "No data provided"}), 400
        
    #     raw_data = pd.DataFrame(raw_data)
    #     feature_set_manager = FeatureSetManager(FEATURE_SETS_FILE)

    #     # Initialize results structure
    #     results = {
    #         "success": False,
    #         "dates": None,
    #         "actual": None,
    #         "predictions": {},
    #         "stats": {},
    #         "errors": {}
    #     }

    #     # Create a shared ModelRunner for efficiency
    #     model_runner = ModelRunner(
    #         raw_data=raw_data,
    #         feature_set_manager=feature_set_manager,
    #         hyperparameters=hyperparameters,
    #         verbose=True
    #     )

    #     successful_models = 0

    #     for model_config in model_list:
    #         model_name = model_config.get("name", "unknown")

    #         try:
    #             print(f"Processing model: {model_name}")

    #             model_results = model_runner.run_single_model(model_config)

    #             # Set shared fields only once (from first successful model)
    #             if results["dates"] is None and model_results.get("dates"):
    #                 results["dates"] = model_results["dates"]
    #             if results["actual"] is None and model_results.get("actual"):
    #                 results["actual"] = model_results["actual"]

    #             # Merge model-specific results
    #             if "predictions" in model_results:
    #                 results["predictions"].update(model_results["predictions"])
    #             if "stats" in model_results:
    #                 results["stats"].update(model_results["stats"])
                
    #             successful_models += 1

    #         except Exception as e:
    #             error_msg = str(e)
    #             print(f"Error processing model {model_name}: {error_msg}")

    #             # Record error but continue with other models
    #             results["predictions"][model_name] = []
    #             results["stats"][model_name] = {"error": error_msg}
    #             results["errors"][model_name] = error_msg

    #     # Determine overall success
    #     results["success"] = successful_models > 0

    #     if successful_models == 0:
    #         return jsonify({
    #             **results,
    #             "error": "All models failed to process"
    #         }), 500
    #     elif successful_models < len(model_list):
    #         return jsonify({
    #             **results,
    #             "warning": f"Only {successful_models}/{len(model_list)} models succeeded"
    #         }), 200
    #     else:
    #         return jsonify(results), 200


    @app.route('/api/run_models_old', methods=['POST'])
    def run_models_old():
        """
        Receive model configs and time series data, run ML models,
        and return predictions & stats as JSON.
        """
        # print("In back-end")

        if request.method == "OPTIONS":
            # Preflight request handled automatically by flask-cors
            return jsonify({"status": "ok"}), 200
        
        data = request.get_json()

        print("Top-level keys received:", list(data.keys()))

        raw_data = data.get("rawData", [])
        model_list = data.get("modelList", [])
        hyperparameters = data.get("hyperparameters")

        if not model_list:
            return jsonify({"error": "No models provided"}), 400
        if not raw_data:
            return jsonify({"error": "No data provided"}), 400

        raw_data = pd.DataFrame(raw_data)
        # print(raw_data.head())
        feature_set_manager = FeatureSetManager(FEATURE_SETS_FILE)

        results = {
            "success": None,
            "dates": None,
            "actual": None,
            "predictions": {},
            "stats": {}
        }

        for model_config in model_list:
            try:

                print(model_config)
                # print(raw_data)
                print(hyperparameters)

                model_results = run_model(model_config, raw_data, feature_set_manager, hyperparameters)

                # print(model_results)

                # Set shared fields only once
                if results["dates"] is None:
                    results["dates"] = model_results["dates"]
                if results["actual"] is None:
                    results["actual"] = model_results["actual"]

                # Merge model-specific results
                results["predictions"].update(model_results["predictions"])
                results["stats"].update(model_results["stats"])

            except Exception as e:
                print(e)
                results["predictions"][model_config["name"]] = []
                results["stats"][model_config["name"]] = {"error": str(e)}

        results["success"] = True

        return jsonify(results)

   

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


def run_model(model_config, raw_data, feature_set_manager, hyperparameters):
    """
    Receive model configs and time series data, run ML models,
    and return predictions & stats as JSON.
    """
    print(f"Running model: {model_config["name"]}")

    # train_val_test_split = hyperparameters.get("train_val_test_split", [0.8, 0.1, 0.1])
    model_name = model_config["model"]
    feature_set_name = model_config.get("featureSet")
    feature_set=feature_set_manager.get_feature_set(feature_set_name)
    
    # normalise = model.get("normalise", False)
    # params = model.get("params", {})
    # forecast_period=model.get("forecastPeriod", 1)
    # input_width=model.get("inputWidth", 1)

    train_ratio, val_ratio, test_ratio = hyperparameters.get("train_val_test_split", [0.8, 0.1, 0.1])

    print("About to create MLData")
    ml_data = DataProcessor(
        raw_data=raw_data, 
        feature_set=feature_set,
        forecast_period=model_config.get("forecastPeriod", 1),
        input_width=model_config.get("inputWidth", 1),
        normalise=model_config.get("normalise", False),
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    print("Created DataProcessor")

    processed_data = ml_data.get_data()
    # window_generator = ml_data.get_window()
    # target = ml_data.get_target()

    # Create model configuration
    model_params = model_config.get("params", {})
    training_params = {k: v for k, v in model_params.items() 
                      if k in ['epochs', 'batch_size', 'optimizer', 'loss']}
    architecture_params = {k: v for k, v in model_params.items() 
                          if k not in ['epochs', 'batch_size', 'optimizer', 'loss']}
    
    config = ModelConfig(
        model_params=architecture_params,
        **training_params
    )

    trainer = ModelTrainer(config, verbose=True)

    input_shape = (processed_data['x_train'].shape[1], processed_data['x_train'].shape[2])
    output_size = processed_data['y_train'].shape[1] if processed_data['y_train'].ndim > 1 else 1

    trainer.train_model(model_name, processed_data, input_shape, output_size)
    results = trainer.evaluate_model(processed_data)


    # Run model
    # model_handler = ModelHandler(
    #     processed_data = processed_data,
    #     params=params,
    #     # window_generator=window_generator,
    #     target=target,
    #     verbose=DEBUG
    # )

    # model_handler.run_keras_model(model_name)

    # results = model_handler.model.model.evaluate(model_handler.x_test, model_handler.y_test)
    # y_test = np.concatenate([labels.numpy() for labels in ml_data.get_window().test.map(lambda x, y: y)])
    # y_pred = model_handler.model.predict(
    #     np.concatenate([x.numpy() for x, _ in ml_data.get_window().test])
    # )
    y_pred = np.array(results['predictions'])
    y_true = np.array(results['actuals'])

    # y_pred = model_handler.model.predict(np.concatenate([x.numpy() for x, _ in ml_data.get_window().test])).flatten()    
    # y_test = np.concatenate([labels.numpy() for labels in ml_data.get_window().test.map(lambda x, y: y)]).flatten()

    # Reverse normalization if needed
    if ml_data.get_normalise():
        target = ml_data.get_target()
        params = ml_data.get_normalisation_params(target)
        # target_params = ml_data.get_normalisation_params(ml_data.get_target())
        if params:
            y_pred = (y_pred * params["std"]) + params["mean"]
            y_true = (y_true * params["std"]) + params["mean"]
            # print("Reversed normalization")
    
    # Dates aligned to test set
    date_series = pd.to_datetime(raw_data["date"], errors="coerce")
    train_len = int(len(raw_data) * train_ratio)
    val_len = int(len(raw_data) * val_ratio)
    start_idx = train_len + val_len
    dates_test = pd.to_datetime(date_series.iloc[start_idx:]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

    # results["dates"] = dates_test
    # results["actual"] = y_test.tolist()
    # results["predictions"][model["name"]] = y_pred.tolist()
    # print(model_handler.get_stats(y_test, y_pred))
    # stats = model_handler.get_stats(y_test, y_pred)
    # results["stats"][model["name"]] = stats
    # # results["stats"][model["name"]] = model_handler.get_stats(y_test, y_pred)
    # print("Stats:", stats)

    # # print("Results:")
    # # print(results)

    results = {
        "dates": dates_test,
        "actual": y_true.tolist(),
        "predictions": {model_config["name"]: y_pred.tolist()},
        "stats": {model_config["name"]: results['metrics']}
    }

    return results


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
        # if category.lower() == "australian":
        #     df = tables[2]  # ASX 200 table index
        #     tickers = df['Code'].astype(str).tolist()
        # elif category.lower() == "us":
        #     df = tables[0]
        #     tickers = df['Symbol'].astype(str).tolist()
        else:
            tickers = df[[]]

        print(f"✅ Found {len(tickers)} tickers for {category}")
        return tickers

    except Exception as e:
        print(f"❌ Error fetching tickers for {category}: {e}")
        return []


# def get_tickers_by_category(category):
#     # Helper function to fetch tickers based on category
#     tickers = []

#     headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

#         if (category == 'australian'):
#             url = "https://en.wikipedia.org/wiki/S%26P/ASX_200"
#             response = requests.get(url, headers=headers)
#             asx200 = pd.read_html(response.text)[2]
#             # asx200 = pd.read_html('https://en.wikipedia.org/wiki/S%26P/ASX_200')[2]
#             tickers = asx200[['Code', 'Company']].to_dict(orient="records")
#     if (category == 'australian'):
#         url = "https://en.wikipedia.org/wiki/S%26P/ASX_200"
#         response = requests.get(url, headers=headers)
#         asx200 = pd.read_html(response.text, flavor='lxml')[2]
#         # asx200 = pd.read_html('https://en.wikipedia.org/wiki/S%26P/ASX_200')[2]
#         tickers = asx200[['Code', 'Company']].to_dict(orient="records")
#     elif (category == 'us'):
#         url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#         response = requests.get(url, headers=headers)
#         sp500 = pd.read_html(response.text)[0]
#         # sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
#         sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
#         tickers = sp500[['Symbol', 'Security']].rename(columns={'Symbol': 'Code', 'Security': 'Company'}).to_dict(orient="records")
#     elif (category == 'fx'):
#         tickers = [{'Code': 'AUDUSD=X', 'Company': 'AUDUSD'}]
#     elif (category == 'crypto'):
#         tickers = [{'Code': 'BTC-USD', 'Company': 'Bitcoin'}]
#     elif (category == 'test'):
#         tickers = [{'Code': 'flat', 'Company': 'Flat Co.'}, {'Code': 'ramp', 'Company': 'Ramp Co.'}, {'Code': 'wave', 'Company': 'Wave Co.'}]
#     return tickers

