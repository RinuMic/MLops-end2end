# tests/test_data_processing.py

import os
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_processing import load_data, split_data

def test_load_data():
    # Test if the data is loaded correctly
    X, y = load_data('../data/diabetes.csv')
    
    # Check that X and y have the correct dimensions
    assert X.shape[0] == y.shape[0], "Mismatch in number of samples between X and y"
    assert X.shape[1] == 8, "The number of features should be 8"

def test_split_data():
    # Test if the data is split correctly
    X, y = load_data('../data/diabetes.csv')
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Check that train and test sets have the correct sizes
    assert len(X_train) + len(X_test) == len(X), "Mismatch in the total number of samples after splitting"
    assert len(y_train) + len(y_test) == len(y), "Mismatch in the total number of labels after splitting"

    # Check that train and test sets are not empty
    assert len(X_train) > 0, "Training set is empty"
    assert len(X_test) > 0, "Test set is empty"
