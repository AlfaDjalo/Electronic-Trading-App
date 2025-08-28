import pandas as pd
import numpy as np
import io
from flask import request, session, jsonify

from stock_data import StockData

# from flask import render_template, request, redirect, url_for, session, jsonify, send_file, send_from_directory  # Add this import for serving files

# API Routes for Front-End
def setup_api_routes(app):
    """
    Set up all routes for the Flask application.

    Args:
        app (Flask): The Flask application instance.
    """

    @app.route("/api/upload-csv", methods=["POST", "OPTIONS"])
    def upload_csv():
        if request.method == "OPTIONS":
            # Preflight request handled automatically by flask-cors
            return jsonify({"status": "ok"}), 200
        
    #     file = request.files["file"]  # or however you’re sending it
    #     # process file...
    #     return jsonify({"message": "CSV uploaded successfully"})



    # @app.route('/api/upload-csv', methods=['POST'])
    # def upload_csv():
        global uploaded_data, data_statistics
        
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not file.filename.endswith('.csv'):
                return jsonify({'error': 'File must be a CSV'}), 400
            
            # Read CSV file
            csv_content = file.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_content))
            
            # Validate data format
            if len(df.columns) < 2:
                return jsonify({'error': 'CSV must have at least 2 columns (date + returns)'}), 400
            
            # Set first column as index (assuming it's dates)
            df.set_index(df.columns[0], inplace=True)
            
            # Validate numeric data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return jsonify({'error': 'No numeric columns found for returns data'}), 400
            
            # Store data globally
            uploaded_data = df[numeric_cols].copy()
            uploaded_data = uploaded_data.reset_index()
            uploaded_data.rename(columns={uploaded_data.columns[0]: "Date"}, inplace=True)
            
            print(uploaded_data.index[0])
            print(uploaded_data.index[-1])

            # Prepare response
            response = {
                'success': True,
                'data_info': {
                    'num_features': len(uploaded_data.columns),
                    'num_observations': len(uploaded_data),
                    'feature_names': uploaded_data.columns.tolist(),
                    'date_range': f"{uploaded_data.index[0]} to {uploaded_data.index[-1]}",
                    'data': uploaded_data.reset_index().to_dict(orient='records')
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            print(f"Upload error: {error_msg}")
            # print(traceback.format_exc())
            return jsonify({'error': error_msg}), 500

    @app.route("/api/features")
    def api_features():
        ticker = request.args.get("ticker")
        stock_data = StockData(session["data_type"], ticker,
                            session["start_date"], session["end_date"], load_data=True)

        raw_data = stock_data.get_data()
        features = stock_data.get_available_fields()

        return jsonify({
            "dates": raw_data.index.strftime("%Y-%m-%d").tolist(),
            "features": features,
            "featuresData": {f: raw_data[f].tolist() for f in features}
        })
