"""
Unit tests for the Flask application.

This module contains tests for the `/predict` endpoint of the Flask application using the 
Flask-Testing library. It includes tests for various scenarios including valid input, missing 
features, invalid features length, and invalid data types.

The tests are organized into the following cases:
- `test_predict_valid_input`: Verifies that the `/predict` endpoint responds correctly to 
  valid input data.
- `test_predict_missing_features`: Verifies that the `/predict` endpoint returns an error 
  when the `features` key is missing in the request.
- `test_predict_invalid_features_length`: Verifies that the `/predict` endpoint returns an 
  error when the length of the `features` list is not 8.
- `test_predict_invalid_features_data_type`: Verifies that the `/predict` endpoint returns 
  an error when the `features` value is not a list of numbers.
"""

import os
import sys
import json
import pytest
from flask_testing import TestCase
from app import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

class AppTestCase(TestCase):
    """
    Unit tests for the Flask application.

    This class contains tests for the `/predict` endpoint of the Flask application using the 
    Flask-Testing library. It includes tests for various scenarios to ensure the robustness of 
    the API, including handling of valid and invalid input data.

    The tests cover the following scenarios:
    - Valid input: Ensures the `/predict` endpoint responds correctly with valid input data.
    - Missing `features` key: Checks that the `/predict` endpoint returns an error when the 
      `features` key is missing from the request.
    - Invalid length of `features` list: Verifies that the `/predict` endpoint returns an error 
      when the length of the `features` list is not equal to 8.
    - Invalid data type for `features`: Validates that the `/predict` endpoint returns an error 
      when the `features` value is not a list of numbers.

    Methods:
        - create_app: Sets up the Flask application for testing.
        - test_predict_valid_input: Tests the `/predict` endpoint with valid input data.
        - test_predict_missing_features: Tests the `/predict` endpoint when the `features` key 
          is missing from the request.
        - test_predict_invalid_features_length: Tests the `/predict` endpoint with an invalid 
          number of features.
        - test_predict_invalid_features_data_type: Tests the `/predict` endpoint with an invalid 
          data type for features.
    """
    def create_app(self):
        """Set up the Flask application for testing."""
        app.config['TESTING'] = True
        return app

    def test_predict_valid_input(self):
        """Test /predict endpoint with valid input."""
        response = self.client.post('/predict', data=json.dumps({
            'features': [6, 148, 72, 35, 0, 33.6, 0.627, 50]
        }), content_type='application/json')
        self.assert200(response)
        self.assertIn('prediction', response.json)

    def test_predict_missing_features(self):
        """Test /predict endpoint with missing features key."""
        response = self.client.post('/predict', data=json.dumps({}), content_type='application/json')
        self.assert400(response)
        self.assertEqual(response.json, {'error': 'Missing features key'})

    def test_predict_invalid_features_length(self):
        """Test /predict endpoint with invalid features length."""
        response = self.client.post('/predict', data=json.dumps({
            'features': [6, 148, 72, 35, 0, 33.6, 0.627]  # Only 7 features provided instead of 8
        }), content_type='application/json')
        self.assert400(response)
        self.assertEqual(response.json, {'error': 'Expecting 8 features'})

    def test_predict_invalid_features_data_type(self):
        """Test /predict endpoint with invalid data type for features."""
        response = self.client.post('/predict', data=json.dumps({
            'features': 'invalid_data'  # Invalid data type
        }), content_type='application/json')
        self.assert400(response)
        self.assertEqual(response.json, {'error': 'Expecting 8 features'})

if __name__ == '__main__':
    pytest.main()
