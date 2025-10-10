def test_run_models_endpoint_success(client, sample_request_data):
    response = client.post('/api/run_models', 
                          json=sample_request_data,
                          headers={'Content-Type': 'application/json'})
    
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] == True