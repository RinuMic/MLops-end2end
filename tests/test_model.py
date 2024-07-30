# tests/test_model.py

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import create_model
from sklearn.ensemble import RandomForestClassifier

def test_create_model_default():
    # Test if the default model is created correctly
    model = create_model()
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier by default"
    assert model.n_estimators == 100, "Default n_estimators should be 100"
    assert model.max_depth is None, "Default max_depth should be None"

def test_create_model_custom():
    # Test if the model is created with custom parameters
    model = create_model(n_estimators=150, max_depth=10)
    assert model.n_estimators == 150, "n_estimators not set correctly"
    assert model.max_depth == 10, "max_depth not set correctly"
