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
import subprocess
import sys
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_processing import load_data, scale_features

# DVC command to pull the data
try:
    subprocess.run(['dvc', 'pull', 'data/diabetes.csv.dvc'], check=True)
except subprocess.CalledProcessError:
    raise RuntimeError("Failed to pull data using DVC")

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
data_path = os.path.join(data_dir, 'diabetes.csv')

# Get the absolute path to the models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../models')

# Define the paths to your model and scaler files
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# MODEL_PATH = '../models/model.pkl'
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError(f'Model file not found: {MODEL_PATH}')

print('###########################:', data_path)
print('#############################:',MODEL_PATH)
print('#############################:',SCALER_PATH)

# Load the scaler
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
else:
    raise FileNotFoundError(f'Scaler file not found: {SCALER_PATH}')


def test_train_and_evaluate():
    # Load the saved model
    # model_path = '../models/model.pkl'
    with open(MODEL_PATH, 'rb') as f:
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
    # model_path = '../models/model.pkl'
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    x, y = load_data(data_path)
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_test_scaled, _ = scale_features(x_test, x_test)
    predictions = model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.6, "Model accuracy is too low"
# tests/test_train2.py

def test_mlflow_artifacts():
    # Check if the scaler and model files exist
    # scaler_path = '../models/scaler.pkl'
    # model_path = '../models/model.pkl'
    assert os.path.exists(SCALER_PATH), "Scaler artifact not found"
    assert os.path.exists(MODEL_PATH), "Model artifact not found"
    # Load and check the scaler and model
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    assert isinstance(scaler, StandardScaler), "Scaler type mismatch"
    assert isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, LogisticRegression, SVC, DecisionTreeClassifier)), "Model type mismatch"
# tests/test_train2.py

def test_model_reloading():
    """
    TEsting model reloading.
    """
    # model_path = '../models/model.pkl'
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    # Check if the model can make predictions
    x, _ = load_data(data_path)
    x_scaled, _ = scale_features(x, x)
    predictions = model.predict(x_scaled)
    assert len(predictions) == x.shape[0], "Model predictions length mismatch"
