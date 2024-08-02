"""
Module docstring for train_mlflow.py.
"""
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
# Set the tracking URI to a local directory
mlflow.set_tracking_uri("http://localhost:5000")

def objective(trial):
    # Load and split the data
    x, y = load_data('data/diabetes.csv')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Scale features
    x_train_scaled, x_test_scaled = scale_features(x_train, x_test)
    # Create a model with hyperparameters suggested by the trial
    model = create_model(trial)
    # Train the model
    model.fit(x_train_scaled, y_train)
    # Evaluate the model
    predictions = model.predict(x_test_scaled)
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
        x, y = load_data('data/diabetes.csv')
        x_scaled, _, scaler = scale_features(x, x, return_scaler=True)  # Scale features on the entire dataset
        model = create_model(trial)
        model.fit(x_scaled, y)
        # Save the scaler and model
        os.makedirs('models', exist_ok=True)
        # Save the scaler
        scaler_path = 'models/scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(scaler_path, "artifacts")
        # Save the trained model
        model_path = 'models/model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_path, "artifacts")
        print(f'Model and scaler saved to {model_path}')
        # Plot and save confusion matrix
        _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        x_test_scaled, _ = scale_features(x_test, x_test)  # Scale features for test set
        predictions = model.predict(x_test_scaled)
        cm = confusion_matrix(y_test, predictions)
        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        # Save confusion matrix plot
        cm_path = 'models/confusion_matrix.png'
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path, "artifacts")

if __name__ == "__main__":
    train_and_evaluate()
