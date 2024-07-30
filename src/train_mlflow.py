# src/train.py

import os
import pickle
import mlflow
import mlflow.sklearn
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    
    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    return accuracy

def train_and_evaluate():
    # Start MLflow run
    with mlflow.start_run() as run:
        # Log experiment parameters
        mlflow.log_param("experiment_name", "Diabetes Model Training")
        
        # Optimize the hyperparameters using Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        # Get the best trial
        trial = study.best_trial
        print(f'Best Model: {trial.params}')
        print(f'Best Accuracy: {trial.value:.2f}')

        # Log the best hyperparameters
        mlflow.log_params(trial.params)
        mlflow.log_metric("best_accuracy", trial.value)

        # Retrain the best model on the entire dataset
        X, y = load_data('../data/diabetes.csv')
        X_scaled, _, scaler = scale_features(X, X, return_scaler=True)  # Scale features on the entire dataset
        model = create_model(trial)
        model.fit(X_scaled, y)

        # Save the scaler and model
        os.makedirs('../models', exist_ok=True)
        
        # Save the scaler
        scaler_path = '../models/scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(scaler_path, "artifacts")

        # Save the trained model
        model_path = '../models/model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_path, "artifacts")
        print(f'Model and scaler saved to {model_path}')

        # Plot and save confusion matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled, _ = scale_features(X_test, X_test)  # Scale features for test set
        predictions = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        # Save confusion matrix plot
        cm_path = '../models/confusion_matrix.png'
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path, "artifacts")

if __name__ == "__main__":
    train_and_evaluate()

