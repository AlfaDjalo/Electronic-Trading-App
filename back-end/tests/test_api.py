import io
import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

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
