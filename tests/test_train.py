"""
Unit tests for model training and evaluation.

This module contains tests for verifying the correctness and functionality of a saved machine learning model.
The tests include:
- Checking if the model matches one of the expected classifier types.
- Evaluating the model's accuracy on a test dataset.
- Verifying the existence and type of model and scaler artifacts.
- Ensuring the model can make predictions after reloading.

The tests assume that the model and scaler artifacts are stored in the '../models' directory and the
dataset is located in '../data'.
"""
# tests/test_train2.py

import os
import sys
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_processing import load_data, scale_features
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def test_train_and_evaluate():
    # Load the saved model
    model_path = '../models/model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    # Define the expected classifier types
    classifier_types = {
        'RandomForest': RandomForestClassifier,
        'GradientBoosting': GradientBoostingClassifier,
        'LogisticRegression': LogisticRegression,
        'SVC': SVC,
        'DecisionTree': DecisionTreeClassifier
    }
    # Check if the model matches one of the expected classifier types
    found_match = False
    for _, classifier_type in classifier_types.items():
        if isinstance(model, classifier_type):
            found_match = True
            break
    assert found_match, "Saved model is not a recognized classifier type"

def test_model_accuracy():
    # Load the best model and evaluate its accuracy
    model_path = '../models/model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    x, y = load_data('../data/diabetes.csv')
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_test_scaled, _ = scale_features(x_test, x_test)
    predictions = model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.6, "Model accuracy is too low"
# tests/test_train2.py

def test_mlflow_artifacts():
    # Check if the scaler and model files exist
    scaler_path = '../models/scaler.pkl'
    model_path = '../models/model.pkl'
    assert os.path.exists(scaler_path), "Scaler artifact not found"
    assert os.path.exists(model_path), "Model artifact not found"
    # Load and check the scaler and model
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    assert isinstance(scaler, StandardScaler), "Scaler type mismatch"
    assert isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, LogisticRegression, SVC, DecisionTreeClassifier)), "Model type mismatch"
# tests/test_train2.py

def test_model_reloading():
    """
    TEsting model reloading.
    """
    model_path = '../models/model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    # Check if the model can make predictions
    x, _ = load_data('../data/diabetes.csv')
    x_scaled, _ = scale_features(x, x)
    predictions = model.predict(x_scaled)
    assert len(predictions) == x.shape[0], "Model predictions length mismatch"
