import os
import sys
import pytest
import json
from flask import Flask
from flask_testing import TestCase
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from app import app

class AppTestCase(TestCase):
    def create_app(self):
        # Set up your Flask application for testing
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

