# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5174"}})  # allow your React origin

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True}), 200

@app.route("/api/upload-csv", methods=["POST"])
def upload_csv():
    # No preflight needed for multipart/form-data; keep it simple.
    if "file" not in request.files:
        return jsonify({"error": "No file field named 'file'"}), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # We won't parse the CSV yet—just prove the path works:
    return jsonify({"message": "CSV received", "filename": f.filename}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
