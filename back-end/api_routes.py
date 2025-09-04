import pandas as pd
import numpy as np
import io
import os
from flask import request, jsonify

from stock_data import StockData
from ml_data import MLData
from model_handler import ModelHandler
from feature_set import FeatureSetManager
# from flask import render_template, request, redirect, url_for, session, jsonify, send_file, send_from_directory  # Add this import for serving files

# FEATURE_SETS_FILE = "c:\\Users\\David\\Projects\\electronic_trading_app\\back-end\\data\\feature_sets.json"
FEATURE_SETS_FILE = os.path.join(os.path.dirname(__file__), 'data\\feature_sets.json')
DEBUG = False

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

        fileName = request.files['fileName']
        if not fileName:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

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




        # date_series = raw_data.iloc[:, 0]

        # try:
        #     date_series = pd.to_datetime(date_series)
        # except Exception:
        #     pass  



        # date_series = raw_data.iloc[:, 0]


        # # Align with test set length (same split as MLData)
        # train_len = int(len(raw_data) * 0.8)
        # dates_test = date_series.iloc[train_len:].tolist()

        # # Format for JSON
        # results["dates"] = dates_test

        # # Store in results
        # results["dates"] = [d.strftime("%Y-%m-%d") if isinstance(d, pd.Timestamp) else str(d) for d in dates_test]

        # results = []
        # predictions = {}
        # errors = {}



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

    
#     # @app.route("/api/process_data", methods=["POST", "OPTIONS"])
#     # def process_data():
#     #     if request.method == "OPTIONS":
#     #         # Preflight request handled automatically by flask-cors
#     #         return jsonify({"status": "ok"}), 200

#     #     raw_data = request.files['rawData']
#     #     feature_set = request.files['featureSet']

#     #     if not raw_data:
#     #         return jsonify({"success": False, "error": "No data available"}), 400

#     #     try:
#     #         processed_data = process_data(raw_data, feature_set)

#     #         response = {
#     #             "success": True,
#     #             "timeSeriesData": raw_data,
#     #         }

#     #         return jsonify(response)
#     #     except Exception as e:
#     #         error_msg = f"Error processing feature set: {str(e)}"
#     #         print(f"Processing error: {error_msg}")
#     #         # print(traceback.format_exc())
#     #         return jsonify({'error': error_msg}), 500


#     @app.route("/api/run_models", methods=["POST", "OPTIONS"])
#     def run_models():
#         if request.method == "OPTIONS":
#             # Preflight request handled automatically by flask-cors
#             return jsonify({"status": "ok"}), 200

#         raw_data = request.files['rawData']
#         model_list = request.files['modelList']
#         hyperparameters = request.files['hyperparameters']

#         if not raw_data:
#             return jsonify({"success": False, "error": "No data available"}), 400

#         if not model_list:
#             return jsonify({"success": False, "error": "No models available"}), 400

#         raw_data_df = pd.DataFrame(raw_data)
#         feature_set_manager = FeatureSetManager(FEATURE_SETS_FILE)
        
#         for model in model_list:
#             try:
#                 feature_set_name = model.get("featureSet")
#                 normalise = model.get("normalise", False)
#                 train_val_test_split = hyperparameters.get("train_val_test_split", [0.8, 0.1, 0.1])
#                 processed_data, target, window_generator = process_data(raw_data_df, feature_set_manager, feature_set_name, normalise, train_val_test_split)
                
#                 # model_name = model_list["model"]
#                 # params = model.get("params", {})

#                 process_model(model, processed_data, target, window_generator)
                


#             response = {
#                 "success": True,
#                 "timeSeriesData": raw_data,
#             }

#             return jsonify(response)
#         except Exception as e:
#             error_msg = f"Error processing feature set: {str(e)}"
#             print(f"Processing error: {error_msg}")
#             # print(traceback.format_exc())
#             return jsonify({'error': error_msg}), 500




#     @app.route('/api/run_models1', methods=['POST'])
#     def run_models1():
#         if request.method == "OPTIONS":
#             # Preflight request handled automatically by flask-cors
#             return jsonify({"status": "ok"}), 200
        
#         data = request.get_json()
#         models = data.get("models", [])
#         data_info = data.get("data", None)

#         if not models or not data_info:
#             return jsonify({"success": False, "error": "No models or data provided"}), 400

#         results_data = run_models(data, models, data_info)

#         # Build DataFrame
#         stock_data_df = pd.DataFrame(data_info["data"])
#         stock_data_df.iloc[:, 0] = pd.to_datetime(stock_data_df.iloc[:, 0], errors="coerce")
#         stock_data_df.rename(columns={stock_data_df.columns[0]: "date"}, inplace=True)

#         date_series = stock_data_df["date"]
#         train_len = int(len(stock_data_df) * 0.8)
#         dates_test = date_series.iloc[train_len:].tolist()

#         results_actual = None
#         predictions = {}
#         stats = {}

#         feature_set_manager = FeatureSetManager(FEATURE_SETS_FILE)

#         for model in models:
#             try:
#                 model_name = model["model"]
#                 feature_set_name = model.get("featureSet")
#                 normalise = model.get("normalise", False)
#                 params = model.get("params", {})

#                 ml_data = MLData(
#                     raw_data=stock_data_df,
#                     train_percentage=0.8,
#                     feature_set=feature_set_manager.get_feature_set(feature_set_name),
#                     normalise=normalise,
#                     verbose=DEBUG
#                 )

#                 model_handler = ModelHandler(
#                     ml_data.get_data(),
#                     params=params,
#                     window_generator=ml_data.get_window(),
#                     target=ml_data.get_target(),
#                     verbose=DEBUG
#                 )

#                 model_handler.run_keras_model(model_name)

#                 y_test = np.concatenate([labels.numpy() for labels in ml_data.get_window().test.map(lambda x, y: y)])
#                 y_pred = model_handler.model.predict(np.concatenate([x.numpy() for x, _ in ml_data.get_window().test]))

#                 # Reverse normalization if needed
#                 if ml_data.get_normalise():
#                     target_params = ml_data.get_normalisation_params(ml_data.get_target())
#                     y_pred = (y_pred * target_params["std"]) + target_params["mean"]
#                     y_test = (y_test * target_params["std"]) + target_params["mean"]

#                 if results_actual is None:
#                     results_actual = y_test.tolist()

#                 predictions[model["name"]] = y_pred.tolist()
#                 stats[model["name"]] = model_handler.get_stats(y_test, y_pred)

#             except Exception as e:
#                 stats[model["name"]] = {"error": str(e)}

#         # Build chartData (date + Actual + predictions)
#         time_series_data = []
#         for i, d in enumerate(dates_test):
#             row = {"date": d}
#             if results_actual is not None:
#                 row["Actual"] = results_actual[i]
#             for model_name, preds in predictions.items():
#                 row[model_name] = preds[i]
#             time_series_data.append(row)

#         # Final response (no seriesNames — frontend infers from keys)
#         response = {
#             "success": True,
#             "timeSeriesData": time_series_data,
#             "stats": stats
#         }
#         return jsonify(response)


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


# def process_data(raw_data_df, feature_set_manager, feature_set_name, normalise, train_val_test_split):
#     # Build DataFrame
#     # df = pd.DataFrame(raw_data)
#     # df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
#     # df.rename(columns={df.columns[0]: "date"}, inplace=True)

#     ml_data = MLData(
#         raw_data=raw_data_df,
#         train_percentage=train_val_test_split[0],
#         val_percentage=train_val_test_split[1]
#         feature_set=feature_set_manager.get_feature_set(feature_set_name),
#         normalise=normalise,
#         verbose=False
#     )


#     processed_data = ml_data.get_data()
#     target = ml_data.get_target()
#     window_generator = ml_data.get_window()

#     return processed_data, target, window_generator



# def process_model(model, processed_data, target, window_generator):
#     try:
#         model_name = model["model"]
#         # feature_set_name = model.get("featureSet")
#         # normalise = model.get("normalise", False)
#         params = model.get("params", {})

#         model_handler = ModelHandler(
#             processed_data,
#             params=params,
#             window_generator=window_generator,
#             target=target,
#             verbose=DEBUG
#         )

#         model_handler.run_keras_model(model_name)

#         y_test = np.concatenate([labels.numpy() for labels in window_generator.test.map(lambda x, y: y)])
#         y_pred = model_handler.model.predict(np.concatenate([x.numpy() for x, _ in window_generator.test]))

#         # Reverse normalization if needed
#         if ml_data.get_normalise():
#             target_params = ml_data.get_normalisation_params(ml_data.get_target())
#             y_pred = (y_pred * target_params["std"]) + target_params["mean"]
#             y_test = (y_test * target_params["std"]) + target_params["mean"]

#         if results_actual is None:
#             results_actual = y_test.tolist()

#         predictions[model["name"]] = y_pred.tolist()
#         stats[model["name"]] = model_handler.get_stats(y_test, y_pred)

#     except Exception as e:
#         stats[model["name"]] = {"error": str(e)}

#     # Build chartData (date + Actual + predictions)
#     time_series_data = []
#     for i, d in enumerate(dates_test):
#         row = {"date": d}
#         if results_actual is not None:
#             row["Actual"] = results_actual[i]
#         for model_name, preds in predictions.items():
#             row[model_name] = preds[i]
#         time_series_data.append(row)




