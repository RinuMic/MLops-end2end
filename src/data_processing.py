# # src/data_processing.py

# import pandas as pd
# from sklearn.model_selection import train_test_split

# def load_data(file_path):
#     """Load the dataset and split it into features and target."""
#     data = pd.read_csv(file_path)
#     X = data.drop('Outcome', axis=1)
#     y = data['Outcome']
#     return X, y

# def split_data(X, y, test_size=0.2, random_state=42):
#     """Split the data into training and testing sets."""
#     return train_test_split(X, y, test_size=test_size, random_state=random_state)


# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load the dataset and split it into features and target."""
    data = pd.read_csv(file_path)
    # Handle missing values: Assume missing values represented by zero
    # data = data.replace(0, np.nan)
    data.fillna(data.mean(), inplace=True)
    # Feature Engineering
    data['BMI*Age'] = data['BMI'] * data['Age']  # Interaction feature
    data['Glucose*Insulin'] = data['Glucose'] * data['Insulin']
    data['Glucose^2'] = data['Glucose'] ** 2
    data['BMI^2'] = data['BMI'] ** 2
    data['LogInsulin'] = np.log1p(data['Insulin'])
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_test, return_scaler=False):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = None

    if return_scaler:
        return X_train_scaled, X_test_scaled, scaler
    else:
        return X_train_scaled, X_test_scaled
