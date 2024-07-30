# src/hyperparameter_tuning.py

import optuna
from data_processing import load_data, split_data
from model import create_model
from sklearn.metrics import accuracy_score

def objective(trial):
    X, y = load_data('../data/diabetes.csv')
    X_train, X_test, y_train, y_test = split_data(X, y)

    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 30)

    model = create_model(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    print("Best Hyperparameters: ", study.best_params)

if __name__ == "__main__":
    main()
