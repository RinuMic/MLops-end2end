# src/model.py

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def create_model(trial):
    """Create a model based on trial suggestions."""
    
    # Define the search space for different classifiers
    classifier_name = trial.suggest_categorical('classifier', ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVC', 'DecisionTree'])
    
    # Model-specific hyperparameters
    if classifier_name == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 2, 30)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
    elif classifier_name == 'GradientBoosting':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        max_depth = trial.suggest_int('max_depth', 2, 10)
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)

    elif classifier_name == 'LogisticRegression':
        C = trial.suggest_float('C', 0.01, 10.0, log=True)
        solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
        model = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)

    elif classifier_name == 'SVC':
        C = trial.suggest_float('C', 0.1, 10.0, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        model = SVC(C=C, kernel=kernel, probability=True, random_state=42)

    elif classifier_name == 'DecisionTree':
        max_depth = trial.suggest_int('max_depth', 2, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)

    return model
