# tests/test_train2.py

import os
import sys
import pickle
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_processing import load_data, split_data, scale_features

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
    for classifier_name, classifier_type in classifier_types.items():
        if isinstance(model, classifier_type):
            found_match = True
            break
    
    assert found_match, "Saved model is not a recognized classifier type"


def test_model_accuracy():
    # Load the best model and evaluate its accuracy
    model_path = '../models/model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    X, y = load_data('../data/diabetes.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled, _ = scale_features(X_test, X_test)
    
    predictions = model.predict(X_test_scaled)
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
    model_path = '../models/model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Check if the model can make predictions
    X, _ = load_data('../data/diabetes.csv')
    X_scaled, _ = scale_features(X, X)
    predictions = model.predict(X_scaled)
    assert len(predictions) == X.shape[0], "Model predictions length mismatch"


