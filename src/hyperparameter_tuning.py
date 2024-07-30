"""
This module provides functions for hyperparameter tuning using Optuna. 
It includes an objective function for model evaluation and a main function 
to run the hyperparameter optimization study.
"""
# src/hyperparameter_tuning.py

import optuna
from sklearn.metrics import accuracy_score
from data_processing import load_data, split_data
from model import create_model

def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.

    Args:
        trial (optuna.trial.Trial): A trial object that suggests hyperparameters.

    Returns:
        float: The accuracy score of the model on the test set.
    """
    x, y = load_data('../data/diabetes.csv')
    x_train, x_test, y_train, y_test = split_data(x, y)

    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 30)

    model = create_model(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy

def main():
    """
    Main function to create an Optuna study and optimize hyperparameters.

    This function sets up the study direction, runs the optimization process,
    and prints the best hyperparameters found.
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    print("Best Hyperparameters: ", study.best_params)

if __name__ == "__main__":
    main()
