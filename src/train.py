# src/train.py

import os
import pickle
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data_processing import load_data, scale_features
from model import create_model

def objective(trial):
    # Load and split the data
    X, y = load_data('../data/diabetes.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Create a model with hyperparameters suggested by the trial
    model = create_model(trial)
    
    # Train the model
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy

def train_and_evaluate():
    # Optimize the hyperparameters using Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Get the best trial
    trial = study.best_trial
    print(f'Best Model: {trial.params}')
    print(f'Best Accuracy: {trial.value:.2f}')

    # Retrain the best model on the entire dataset
    X, y = load_data('../data/diabetes.csv')
    X_scaled, _, scaler = scale_features(X, X, return_scaler=True)  # Scale features on the entire dataset
    model = create_model(trial)
    model.fit(X_scaled, y)

    # Save the scaler and PCA
    os.makedirs('../models', exist_ok=True)
    with open('../models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Ensure models directory exists
    os.makedirs('../models', exist_ok=True)
        # Save the trained model
    model_path = '../models/model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to {model_path}')

if __name__ == "__main__":
    train_and_evaluate()

