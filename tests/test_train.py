# tests/test_train.py

import os
import sys
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from train import train_and_evaluate
from sklearn.ensemble import RandomForestClassifier

def test_train_and_evaluate():
    # Test if the model is trained and saved correctly
    train_and_evaluate('../data/diabetes.csv')

    # Check if the model file exists
    assert os.path.exists('../models/model.pkl'), "Model file was not created"

    # Load the model and check its type
    with open('../models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    assert isinstance(model, RandomForestClassifier), "Saved model is not a RandomForestClassifier"
