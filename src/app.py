"""
This module initializes the Flask application and sets up the prediction endpoint.
"""

# src/app.py

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from

# Initialize Flask application
app = Flask(__name__)

# Initialize Swagger
swagger = Swagger(app)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/model.pkl')
# MODEL_PATH = '../models/model.pkl'
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError(f'Model file not found: {MODEL_PATH}')

# Initialize scaler and PCA (assuming you saved the scaler and PCA during training)
with open('../models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def feature_engineering(data):
    """Apply feature engineering steps to the data."""
    # Handle missing values: Assume missing values represented by zero
    # data = data.replace(0, np.nan)
    data.fillna(data.mean(), inplace=True)
    # Feature Engineering
    data['BMI*Age'] = data['BMI'] * data['Age']  # Interaction feature
    data['Glucose*Insulin'] = data['Glucose'] * data['Insulin']
    data['Glucose^2'] = data['Glucose'] ** 2
    data['BMI^2'] = data['BMI'] ** 2
    data['LogInsulin'] = np.log1p(data['Insulin'])
    return data

@app.route('/predict', methods=['POST'])
@swag_from({
    'summary': 'Predict diabetes from input features',
    'description': 'Provide patient health metrics to predict diabetes likelihood.',
    'tags': ['Prediction Endpoint'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'features': {
                        'type': 'array',
                        'items': {'type': 'number'},
                        'example': [6, 148, 72, 35, 0, 33.6, 0.627, 50],  # Default example
                    }
                },
                'required': ['features']
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Prediction result',
            'schema': {
                'type': 'object',
                'properties': {
                    'prediction': {
                        'type': 'integer',
                        'description': 'Diabetes prediction (0: No, 1: Yes)',
                        'example': 1
                    }
                }
            }
        }
    }
})
def predict():
    """Predict diabetes from input features"""
    data = request.get_json(force=True)
    if 'features' not in data:
        return jsonify({'error': 'Missing features key'}), 400
    features = data['features']
    if len(features) != 8:
        return jsonify({'error': 'Expecting 8 features'}), 400
    # Convert to DataFrame
    features_df = pd.DataFrame([features], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    # Apply feature engineering
    features_df = feature_engineering(features_df)
    # Extract features for prediction
    features_array = features_df.values
    # Apply the same scaling and PCA as during training
    features_scaled = scaler.transform(features_array)
    # Make prediction
    prediction = model.predict(features_scaled)
    # prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
