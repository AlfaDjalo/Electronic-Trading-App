import pandas as pd
import numpy as np
import io
import os
from flask import request, jsonify
import json

from stock_data import StockData
from ml_data import MLData
from model_handler import ModelHandler
from feature_set import FeatureSetManager
# from flask import render_template, request, redirect, url_for, session, jsonify, send_file, send_from_directory  # Add this import for serving files

# FEATURE_SETS_FILE = "c:\\Users\\David\\Projects\\electronic_trading_app\\back-end\\data\\feature_sets.json"
DEBUG = True
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
        Receive model configs and time series data, run ML models,
        and return predictions & stats as JSON.
        """
        # print("In back-end")

        if request.method == "OPTIONS":
            # Preflight request handled automatically by flask-cors
            return jsonify({"status": "ok"}), 200
        
        data = request.get_json()

        raw_data = data.get("rawData", [])
        model_list = data.get("modelList", [])
        hyperparameters = data.get("hyperparameters")

        if not model_list:
            return jsonify({"error": "No models provided"}), 400
        if not raw_data:
            return jsonify({"error": "No data provided"}), 400

        raw_data = pd.DataFrame(raw_data)
        print(raw_data.head())
        feature_set_manager = FeatureSetManager(FEATURE_SETS_FILE)

        results = {
            "success": None,
            "dates": None,
            "actual": None,
            "predictions": {},
            "stats": {}
        }

        for model in model_list:
            try:

                # print(model)
                # print(raw_data)
                # print(hyperparameters)

                model_results = run_model(model, raw_data, feature_set_manager, hyperparameters)

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
                results["predictions"][model["name"]] = []
                results["stats"][model["name"]] = {"error": str(e)}

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


def run_model(model, raw_data, feature_set_manager, hyperparameters):
    """
    Receive model configs and time series data, run ML models,
    and return predictions & stats as JSON.
    """
    print("In run_model")
    train_val_test_split = hyperparameters.get("train_val_test_split", [0.8, 0.1, 0.1])

    results = {
        "dates": None,
        "actual": None,
        "predictions": {},
        "stats": {}
    }

    model_name = model["model"]
    feature_set_name = model.get("featureSet")
    normalise = model.get("normalise", False)
    params = model.get("params", {})
    print("About to create MLData")
    ml_data = MLData(
        raw_data=raw_data, 
        train_percentage=train_val_test_split[0],
        val_percentage=train_val_test_split[1],
        feature_set=feature_set_manager.get_feature_set(feature_set_name),
        normalise=normalise,
        verbose=DEBUG
    )

    print("Created MLData")
    # print(ml_data.get_data())
    # print(params)
    # print(ml_data.get_window())
    # print(ml_data.get_target())
    # print(model_name)

    # Run model
    model_handler = ModelHandler(
        ml_data.get_data(),
        params=params,
        window_generator=ml_data.get_window(),
        target=ml_data.get_target(),
        verbose=DEBUG
    )

    model_handler.run_keras_model(model_name)
    # y_test = np.concatenate([labels.numpy() for labels in ml_data.get_window().test.map(lambda x, y: y)])
    # y_pred = model_handler.model.predict(
    #     np.concatenate([x.numpy() for x, _ in ml_data.get_window().test])
    # )
    y_test = np.concatenate([labels.numpy() for labels in ml_data.get_window().test.map(lambda x, y: y)]).flatten()
    y_pred = model_handler.model.predict(np.concatenate([x.numpy() for x, _ in ml_data.get_window().test])).flatten()    

    # Reverse normalization if needed
    if ml_data.get_normalise():
        target_params = ml_data.get_normalisation_params(ml_data.get_target())
        y_pred = (y_pred * target_params["std"]) + target_params["mean"]
        y_test = (y_test * target_params["std"]) + target_params["mean"]
        print("Reversed normalization")
    
    # Dates aligned to test set
    date_series = pd.to_datetime(raw_data["date"], errors="coerce")
    train_len = int(len(raw_data) * train_val_test_split[0])
    val_len = int(len(raw_data) * train_val_test_split[1])
    start_idx = train_len + val_len
    # dates_test = date_series.iloc[start_idx:].dt.strftime("%Y-%m-%d").tolist()
    # dates_test = date_series.iloc[start_idx:].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    dates_test = pd.to_datetime(date_series.iloc[start_idx:]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

    results["dates"] = dates_test
    results["actual"] = y_test.tolist()
    results["predictions"][model["name"]] = y_pred.tolist()
    # print(model_handler.get_stats(y_test, y_pred))
    # results["stats"][model["name"]] = model_handler.get_stats(y_test, y_pred)
    # print("Set stats")

    # print("Results:")
    # print(results)

    return results

