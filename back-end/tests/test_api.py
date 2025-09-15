import io
import pandas as pd
import pytest
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

# Test /api/upload_data
def test_upload_data_no_file(client):
    response = client.post("/api/upload_data", data={})
    json_data = response.get_json(force=True)
    assert response.status_code == 400
    assert json_data["success"] is False
    assert "No file uploaded" in json_data["error"]

def test_upload_data_valid_csv(client, tmp_path):
    # Create sample CSV
    csv_content = "date, price\n2023-01-01, 100\n2023-01-02, 105\n"
    csv_file = io.BytesIO(csv_content.encode("utf-8"))
    data = {"fileName": (csv_file, "test.csv")}

    response = client.post("/api/upload_data", data=data, content_type="multipart/form-data")
    json_data = response.get_json()

    assert response.status_code == 200
    assert json_data["success"] is True
    assert "timeSeriesData" in json_data
    assert json_data["timeSeriesData"][0]["date"] == "2023-01-01 00:00:00"

# Test /api/functions
def test_functions_endpoint(client):
    response = client.get("/api/functions")
    json_data = response.get_json()

    assert response.status_code == 200
    assert isinstance(json_data, list)
    assert "create_lag" in json_data  # Example, depends on your function list


# Test /api/feature_sets
def test_feature_sets_endpoint(client, tmp_path, mocker):
    # Mock FEATURE_SETS_FILE path to tmp
    feature_sets_file = tmp_path / "feature_sets.json"
    mocker.patch("api_routes.FEATURE_SETS_FILE", str(feature_sets_file))

    # Write a fake feature set
    fake_sets = {
        "default": {
            "data_type": "daily",
            "features": [
                {"id": 1, "name": "price_raw", "input_data_fields": ["price"], "function": "raw_data", "function_parameters": {}}
            ],
            "target": {
                "id": 2, "name": "target", "input_data_fields": ["price"], "function": "sma", "function_parameters": {"window": 3}
            }
        }
    }
    feature_sets_file.write_text(json.dumps(fake_sets, indent=2))

    response = client.get("/api/feature_sets")
    json_data = response.get_json()

    assert response.status_code == 200
    assert "default" in json_data
    assert "features" in json_data["default"]
    assert "target" in json_data["default"]


# Test /api/save_feature_set
def test_save_feature_set_endpoint(client, tmp_path, mocker):
    # Mock FEATURE_SETS_FILE path to tmp
    feature_sets_file = tmp_path / "feature_sets.json"
    mocker.patch("api_routes.FEATURE_SETS_FILE", str(feature_sets_file))

    # Initial file contents
    feature_sets_file.write_text(json.dumps({}, indent=2))

    new_feature_set = {
        "data_type": "daily",
        "features": [
            {"id": 1, "name": "close_raw", "input_data_fields": ["close"], "function": "raw_data", "function_parameters": {}}
        ],
        "target": {
            "id": 2, "name": "target", "input_data_fields": ["close"], "function": "sma", "function_parameters": {"window": 5}
        }
    }

    response = client.post("/api/save_feature_set", json={
        "name": "test_set",
        "feature_set": new_feature_set
    })

    json_data = response.get_json()
    assert response.status_code == 200
    assert json_data["status"] == "ok"
    assert "saved" in json_data["message"].lower()

    # Ensure file updated
    saved_data = json.loads(feature_sets_file.read_text())
    assert "test_set" in saved_data
    assert saved_data["test_set"]["data_type"] == "daily"


def test_save_feature_set_missing_name_or_content(client):
    response = client.post("/api/save_feature_set", json={"name": "incomplete"})
    assert response.status_code == 400
    json_data = response.get_json()
    assert "error" in json_data

def test_save_feature_set_overwrites_existing(client, tmp_path, mocker):
    # Mock FEATURE_SETS_FILE path to tmp
    feature_sets_file = tmp_path / "feature_sets.json"
    mocker.patch("api_routes.FEATURE_SETS_FILE", str(feature_sets_file))

    # Write initial feature set
    initial_sets = {
        "my_set": {
            "data_type": "daily",
            "features": [
                {"id": 1, "name": "price_raw", "input_data_fields": ["price"], "function": "raw_data", "function_parameters": {}}
            ],
            "target": {
                "id": 2, "name": "target", "input_data_fields": ["price"], "function": "sma", "function_parameters": {"window": 3}
            }
        }
    }
    feature_sets_file.write_text(json.dumps(initial_sets, indent=2))

    # New content with modified target
    updated_set = {
        "data_type": "daily",
        "features": [
            {"id": 1, "name": "price_raw", "input_data_fields": ["price"], "function": "raw_data", "function_parameters": {}}
        ],
        "target": {
            "id": 2, "name": "target", "input_data_fields": ["price"], "function": "ema", "function_parameters": {"span": 10}
        }
    }

    response = client.post("/api/save_feature_set", json={
        "name": "my_set",
        "feature_set": updated_set
    })

    json_data = response.get_json()
    assert response.status_code == 200
    assert json_data["status"] == "ok"

    # Ensure overwrite happened
    saved_data = json.loads(feature_sets_file.read_text())
    assert "my_set" in saved_data
    assert saved_data["my_set"]["target"]["function"] == "ema"
    assert saved_data["my_set"]["target"]["function_parameters"]["span"] == 10


# Test /api/run_models
def test_run_models_missing_data(client):
    response = client.post("/api/run_models", json={
        "rawData": [],
        "modelList": [{"name": "dummy", "model": "baseline"}]
    })
    assert response.status_code == 400
    assert "No data provided" in response.get_json()["error"]

def test_run_models_missing_models(client):
    response = client.post("/api/run_models", json={"modelList": []})
    assert response.status_code == 400
    assert "No models provided" in response.get_json()["error"]

def test_run_models_with_mock(mocker, client):
    # Fake raw data
    raw_data = [
        {"date": "2023-01-01 00:00:00", "price": 100},
        {"date": "2023-01-02 00:00:00", "price": 105},
    ]
    model_list = [{"name": "baseline-Test", "model": "baseline"}]

    # Mock run_model to avoid heavy ML
    mocker.patch("api_routes.run_model", return_value={
        "dates": ["2023-01-02 00:00:00"],
        "actual": [105],
        "predictions": {"baseline-Test": [106]},
        "stats": {"baseline-Test": {"mse": 1.0}},
    })

    response = client.post("/api/run_models", json={
        "rawData": raw_data,
        "modelList": model_list,
        "hyperparameters": {"train_val_test_split": [0.5, 0.5, 0.0]}
    })

    json_data = response.get_json()
    assert response.status_code == 200
    assert json_data["success"] is True
    assert "predictions" in json_data
    assert "baseline-Test" in json_data["predictions"]
