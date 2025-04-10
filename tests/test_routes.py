import pytest
import sys
import os

# Add the project directory to the Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

@pytest.fixture
def client():
    """Set up a test client for the Flask app."""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test the home page loads successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to the Trading App" in response.data

def test_create_comparison(client):
    """Test creating a new comparison."""
    with client.session_transaction() as session:
        session['ticker'] = 'AAPL'
        session['start_date'] = '2022-01-01'
        session['end_date'] = '2022-12-31'
    response = client.post('/create_comparison', data={
        'model': 'LinearRegression',
        'lag_period': 3,
        'forecast_period': 1
    })
    assert response.status_code == 200
    assert b"Manage Comparisons" in response.data

def test_set_parameters(client):
    """Test setting parameters for a comparison."""
    with client.session_transaction() as session:
        session['comparisons'] = [{
            'name': 'Comparison_1',
            'model': 'LinearRegression',
            'params': {
                'forward_projection_days': {'type': 'integer', 'default': 1, 'value': 1, 'min': 1, 'max': 5}
            }
        }]
    response = client.post('/set_parameters/0', data={
        'forward_projection_days': 2
    })
    assert response.status_code == 302  # Redirect after successful update

def test_run_comparisons(client):
    """Test running comparisons."""
    with client.session_transaction() as session:
        session['ticker'] = 'AAPL'
        session['start_date'] = '2022-01-01'
        session['end_date'] = '2022-12-31'
        session['comparisons'] = [{
            'name': 'Comparison_1',
            'model': 'LinearRegression',
            'params': {
                'forward_projection_days': {'type': 'integer', 'default': 1, 'value': 1, 'min': 1, 'max': 5}
            }
        }]
    response = client.post('/run_comparisons')
    assert response.status_code == 302  # Redirect to results page

def test_comparison_results(client):
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
    assert b"Comparison Results for AAPL" in response.data
