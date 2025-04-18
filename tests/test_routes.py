import pytest
from flask import Flask, session
from routes import setup_routes

@pytest.fixture
def client():
    """Set up a test client for the Flask app."""
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.secret_key = 'test_secret_key'
    setup_routes(app)
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """Test the index route."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"index.html" in response.data

def test_create_comparison_route(client):
    """Test creating a new comparison."""
    with client.session_transaction() as session:
        session['ticker'] = 'AAPL'
        session['start_date'] = '2022-01-01'
        session['end_date'] = '2022-12-31'
    response = client.post('/create_comparison', data={
        'model': 'LinearRegression',
        'feature_set': 'OHLCV',
        'normalise': 'on'
    })
    assert response.status_code == 200
    assert b"comparison_page.html" in response.data

def test_set_parameters_route(client):
    """Test setting parameters for a comparison."""
    with client.session_transaction() as session:
        session['comparisons'] = [{
            'name': 'Comparison_1',
            'model': 'LinearRegression',
            'params': {'param1': {'type': 'integer', 'value': 1}}
        }]
    response = client.post('/set_parameters/0', data={'param1': '2'})
    assert response.status_code == 302  # Redirect after successful update

def test_edit_comparison_route(client):
    """Test editing an existing comparison."""
    with client.session_transaction() as session:
        session['comparisons'] = [{
            'name': 'Comparison_1',
            'model': 'LinearRegression',
            'feature_set': 'OHLCV',
            'normalise': False
        }]
    response = client.post('/edit_comparison/0', data={
        'name': 'Updated_Comparison',
        'model': 'LSTM',
        'feature_set': 'OHLCV',
        'normalise': 'on'
    })
    assert response.status_code == 302  # Redirect after successful update

def test_delete_comparison_route(client):
    """Test deleting a comparison."""
    with client.session_transaction() as session:
        session['comparisons'] = [{'name': 'Comparison_1'}]
    response = client.post('/delete_comparison/0')
    assert response.status_code == 302  # Redirect after deletion

def test_clear_comparisons_route(client):
    """Test clearing all comparisons."""
    with client.session_transaction() as session:
        session['comparisons'] = [{'name': 'Comparison_1'}]
    response = client.post('/clear_comparisons')
    assert response.status_code == 302  # Redirect after clearing

def test_run_comparisons_route(client):
    """Test running comparisons."""
    with client.session_transaction() as session:
        session['ticker'] = 'AAPL'
        session['start_date'] = '2022-01-01'
        session['end_date'] = '2022-12-31'
        session['comparisons'] = [{
            'name': 'Comparison_1',
            'model': 'LinearRegression',
            'params': {'param1': {'type': 'integer', 'value': 1}}
        }]
    response = client.post('/run_comparisons')
    assert response.status_code == 302  # Redirect to results page

def test_comparison_results_route(client):
    """Test the comparison results page."""
    with client.session_transaction() as session:
        session['comparison_results'] = {
            'results': [{'name': 'Comparison_1', 'model': 'LinearRegression', 'stats': 'N/A', 'error': 'N/A'}],
            'selected_ticker': 'AAPL'
        }
        session['chart_paths'] = {
            'prediction_chart': 'temp_charts/prediction_chart.png',
            'error_chart': 'temp_charts/error_chart.png'
        }
    response = client.get('/comparison_results')
    assert response.status_code == 200
    assert b"comparison_results.html" in response.data

def test_manage_feature_sets_route(client):
    """Test the manage_feature_sets route."""
    response = client.get('/manage_feature_sets')
    assert response.status_code == 200
    assert b"manage_feature_sets.html" in response.data

def test_set_ticker_and_dates_route(client):
    """Test setting ticker and dates."""
    response = client.post('/set_ticker_and_dates', data={
        'category': 'us',
        'ticker': 'AAPL',
        'start_date': '2022-01-01',
        'end_date': '2022-12-31'
    })
    assert response.status_code == 302  # Redirect after successful update

def test_get_tickers_route(client):
    """Test fetching tickers by category."""
    response = client.get('/get_tickers/us')
    assert response.status_code == 200
    assert b"AAPL" in response.data
