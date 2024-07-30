# tests/test_flask_app.py

import os
import sys
import pytest
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict(client):
    # Test the predict endpoint
    response = client.post('/predict', data=json.dumps({
        'features': [5, 116, 74, 0, 0, 25.6, 0.201, 30]
    }), content_type='application/json')
    
    # Check if the response is successful
    assert response.status_code == 200, "Request to /predict failed"

    # Check if the prediction is in the response
    data = response.get_json()
    assert 'prediction' in data, "No prediction found in the response"
