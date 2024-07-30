# tests/test_hyperparameter_tuning.py

import os
import sys
from io import StringIO
import pytest
import optuna
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from hyperparameter_tuning import objective, main
from model import create_model

# Mock function for create_model to avoid actual model creation
def mock_create_model(n_estimators, max_depth):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

@pytest.fixture
def sample_data():
    """Generate a sample dataset for testing."""
    data = load_diabetes()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def test_objective(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    
    # Mock the create_model function
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr('hyperparameter_tuning.create_model', mock_create_model)
        
        # Test objective function with sample hyperparameters
        trial = optuna.trial.FixedTrial({
            'n_estimators': 100,
            'max_depth': 10
        })
        
        accuracy = objective(trial)
        assert isinstance(accuracy, float), "Objective function did not return a float"
        assert 0.0 <= accuracy <= 1.0, "Objective function returned invalid accuracy"

def test_hyperparameter_tuning():
    # Capture the output of the main function
    with pytest.MonkeyPatch.context() as mp:
        fake_out = StringIO()
        mp.setattr('sys.stdout', fake_out)
        
        # Run the hyperparameter tuning main function
        main()
        
        # Check the printed output
        output = fake_out.getvalue()
        assert "Best Hyperparameters:" in output, "Best hyperparameters were not printed"
        
        # Verify Optuna study results if possible
        # Replace with actual logic to load the study if needed
        study = optuna.load_study(study_name="study_name", storage="sqlite:///example.db")
        assert 'n_estimators' in study.best_params, "Best hyperparameters missing 'n_estimators'"
        assert 'max_depth' in study.best_params, "Best hyperparameters missing 'max_depth'"
        assert isinstance(study.best_params['n_estimators'], int), "'n_estimators' is not an integer"
        assert isinstance(study.best_params['max_depth'], int), "'max_depth' is not an integer"
