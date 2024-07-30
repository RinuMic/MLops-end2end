"""
Unit tests for model creation functionality.

This module contains tests for the `create_model` function in the `model` module. It verifies that
the correct machine learning model is created based on the provided trial suggestions. The tests
cover various models including:
- RandomForestClassifier
- GradientBoostingClassifier
- LogisticRegression
- SVC
- DecisionTreeClassifier
"""
# tests/test_model.py

import os
import sys
from unittest.mock import MagicMock
import pytest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from model import create_model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def test_create_model_random_forest():
    """Test model creation for RandomForestClassifier."""
    trial = MagicMock()
    trial.suggest_categorical.return_value = 'RandomForest'
    trial.suggest_int.side_effect = [100, 10]  # Mock n_estimators and max_depth
    model = create_model(trial)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
    assert model.max_depth == 10

def test_create_model_gradient_boosting():
    """Test model creation for GradientBoostingClassifier."""
    trial = MagicMock()
    trial.suggest_categorical.return_value = 'GradientBoosting'
    trial.suggest_int.side_effect = [100, 3]  # Mock n_estimators and max_depth
    trial.suggest_float.side_effect = [0.1]  # Mock learning_rate
    model = create_model(trial)
    assert isinstance(model, GradientBoostingClassifier)
    assert model.n_estimators == 100
    assert model.learning_rate == 0.1
    assert model.max_depth == 3

def test_create_model_logistic_regression():
    """Test model creation for LogisticRegression."""
    trial = MagicMock()
    trial.suggest_categorical.side_effect = ['LogisticRegression', 'liblinear']
    trial.suggest_float.return_value = 0.1  # Mock C
    model = create_model(trial)
    assert isinstance(model, LogisticRegression)
    assert model.C == 0.1
    assert model.solver == 'liblinear'

def test_create_model_svc():
    """Test model creation for SVC."""
    trial = MagicMock()
    trial.suggest_categorical.side_effect = ['SVC', 'linear']
    trial.suggest_float.return_value = 0.1  # Mock C
    model = create_model(trial)
    assert isinstance(model, SVC)
    assert model.C == 0.1
    assert model.kernel == 'linear'

def test_create_model_decision_tree():
    """Test model creation for DecisionTreeClassifier."""
    trial = MagicMock()
    trial.suggest_categorical.return_value = 'DecisionTree'
    trial.suggest_int.side_effect = [10, 5]  # Mock max_depth and min_samples_split
    model = create_model(trial)
    assert isinstance(model, DecisionTreeClassifier)
    assert model.max_depth == 10
    assert model.min_samples_split == 5

if __name__ == '__main__':
    pytest.main()
