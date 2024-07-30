"""
Unit tests for data processing functions.

This module contains tests for the functions in the `data_processing` module, including:
- `load_data`: Verifies correct loading and feature engineering from the dataset.
- `split_data`: Checks the splitting of the dataset into training and testing sets.
- `scale_features`: Ensures that features are scaled correctly using StandardScaler.

The module also includes tests for handling missing values in the dataset.
"""
# tests/test_data_processing.py

import os
import sys
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_processing import load_data, split_data, scale_features
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture(scope="module")
def sample_data():
    """Create a sample dataset for testing."""
    # Create a sample dataframe with the same columns expected by your functions
    data = {
        'Pregnancies': [6, 1, 8, 1, 0],
        'Glucose': [148, 85, 183, 89, 137],
        'BloodPressure': [72, 66, 64, 66, 40],
        'SkinThickness': [35, 29, 0, 23, 35],
        'Insulin': [0, 0, 0, 94, 168],
        'BMI': [33.6, 26.6, 23.3, 28.1, 43.1],
        'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288],
        'Age': [50, 31, 32, 21, 33],
        'Outcome': [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    df.to_csv('../data/sample_diabetes.csv', index=False)
    return '../data/sample_diabetes.csv'

def test_load_data(sample_data):
    """Test the load_data function.

    Verifies that features and target are correctly extracted and engineered.
    Ensures the presence of all expected features in the dataset.
    """
    x, y = load_data(sample_data)
    # Check the shape of X and y
    assert x.shape == (5, 13), "Feature shape mismatch"
    assert y.shape == (5,), "Target shape mismatch"
    # List of expected feature columns
    expected_features = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
        'BMI*Age', 'Glucose*Insulin', 'Glucose^2',
        'BMI^2', 'LogInsulin'
    ]
    # Verify that each expected feature is present in x.columns
    for feature in expected_features:
        assert feature in x.columns, f"Feature '{feature}' not found"
    # Optionally, check for the presence of some features to ensure they're generated
    assert 'Pregnancies' in x.columns, "Feature 'Pregnancies' not found"
    assert 'Glucose' in x.columns, "Feature 'Glucose' not found"
    assert 'BloodPressure' in x.columns, "Feature 'BloodPressure' not found"
    assert 'SkinThickness' in x.columns, "Feature 'SkinThickness' not found"
    assert 'Insulin' in x.columns, "Feature 'Insulin' not found"
    assert 'BMI' in x.columns, "Feature 'BMI' not found"
    assert 'DiabetesPedigreeFunction' in x.columns, "Feature 'DiabetesPedigreeFunction' not found"
    assert 'Age' in x.columns, "Feature 'Age' not found"
    assert 'BMI*Age' in x.columns, "Feature 'BMI*Age' not found"
    assert 'Glucose*Insulin' in x.columns, "Feature 'Glucose*Insulin' not found"
    assert 'Glucose^2' in x.columns, "Feature 'Glucose^2' not found"
    assert 'BMI^2' in x.columns, "Feature 'BMI^2' not found"
    assert 'LogInsulin' in x.columns, "Feature 'LogInsulin' not found"

def test_split_data(sample_data):
    """Test the split_data function.

    Verifies that the data is split correctly into training and testing sets,
    and checks that neither set is empty and the total length matches the original dataset.
    """
    X, y = load_data(sample_data)
    x_train, x_test, _, _ = split_data(X, y)
    assert len(x_train) > 0, "Training set is empty"
    assert len(x_test) > 0, "Testing set is empty"
    assert len(x_train) + len(x_test) == len(X), "Data splitting mismatch"

def test_scale_features(sample_data):
    """Test the scale_features function.

    Verifies that the features are scaled correctly using StandardScaler,
    and checks if the scaling transforms the features to have a mean close to 0 and std close to 1.
    """
    X, _ = load_data(sample_data)
    x_train, _, scaler = scale_features(X, None, return_scaler=True)
    assert isinstance(scaler, StandardScaler), "Scaler was not returned correctly"
    # Check if scaling transformed the features (mean close to 0 and std close to 1)
    assert np.allclose(x_train.mean(axis=0), 0, atol=1e-2), "Scaled features' mean is not close to 0"
    assert np.allclose(x_train.std(axis=0), 1, atol=1e-2), "Scaled features' std is not close to 1"

def test_load_data_missing_values(sample_data):
    """Test load_data function's handling of missing values.

    Modifies the sample data to include NaN values, then verifies that missing values
    are correctly imputed and no NaNs are present in the dataset.
    """
    # Modify the sample data to include NaN values for testing
    df = pd.read_csv(sample_data)
    df.loc[0, 'BMI'] = np.nan
    df.to_csv(sample_data, index=False)
    X, y = load_data(sample_data)
    assert X['BMI'].isna().sum() == 0, "Missing values in 'BMI' column after imputation"

if __name__ == "__main__":
    pytest.main()
