from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
ROOT = Path(__file__).parent.parent
mlflow.set_tracking_uri(f"sqlite:///{ROOT}/mlflow.db")
mlflow.set_experiment("churn-prediction")


ROOT = Path(__file__).parent.parent

def load_data():
    df = pd.read_csv(ROOT / "data" / "preprocessed.csv")
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def log_metrics(y_test, y_pred):
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_churn": f1_score(y_test, y_pred),
        "recall_churn": recall_score(y_test, y_pred),
        "precision_churn": precision_score(y_test, y_pred),
    })
    print(classification_report(y_test, y_pred))


def train_random_forest(X_train, X_test, y_train, y_test):
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced"],
    }
    base = RandomForestClassifier(random_state=RANDOM_STATE)
    search = RandomizedSearchCV(base, param_dist, n_iter=20, cv=5, scoring="f1", n_jobs=-1, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name="random_forest"):
        search.fit(X_train, y_train)
        best = search.best_estimator_
        mlflow.log_params(search.best_params_)
        mlflow.sklearn.log_model(best, "model")
        log_metrics(y_test, best.predict(X_test))
        print(f"Best RF params: {search.best_params_}\n")
    return best


def train_xgboost(X_train, X_test, y_train, y_test):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "scale_pos_weight": [scale_pos_weight],
    }
    base = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0)
    search = RandomizedSearchCV(base, param_dist, n_iter=20, cv=5, scoring="f1", n_jobs=-1, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name="xgboost"):
        search.fit(X_train, y_train)
        best = search.best_estimator_
        mlflow.log_params(search.best_params_)
        mlflow.sklearn.log_model(best, "model")
        log_metrics(y_test, best.predict(X_test))
        print(f"Best XGB params: {search.best_params_}\n")
    return best


def train_lightgbm(X_train, X_test, y_train, y_test):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [20, 31, 50, 80],
        "subsample": [0.7, 0.8, 1.0],
        "scale_pos_weight": [scale_pos_weight],
    }
    base = LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)
    search = RandomizedSearchCV(base, param_dist, n_iter=20, cv=5, scoring="f1", n_jobs=-1, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name="lightgbm"):
        search.fit(X_train, y_train)
        best = search.best_estimator_
        mlflow.log_params(search.best_params_)
        mlflow.sklearn.log_model(best, "model")
        log_metrics(y_test, best.predict(X_test))
        print(f"Best LGBM params: {search.best_params_}\n")
    return best


def train_ensemble(rf, xgb, lgbm, X_train, X_test, y_train, y_test):
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        voting="soft",
    )
    with mlflow.start_run(run_name="ensemble_voting"):
        ensemble.fit(X_train, y_train)
        mlflow.sklearn.log_model(ensemble, "model")
        log_metrics(y_test, ensemble.predict(X_test))
    return ensemble


def main():
    X_train, X_test, y_train, y_test = load_data()

    # Apply SMOTE to training set to handle class imbalance
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE — class distribution: {y_train_res.value_counts().to_dict()}\n")

    print("=== Random Forest ===")
    rf = train_random_forest(X_train_res, X_test, y_train_res, y_test)

    print("=== XGBoost ===")
    xgb = train_xgboost(X_train_res, X_test, y_train_res, y_test)

    print("=== LightGBM ===")
    lgbm = train_lightgbm(X_train_res, X_test, y_train_res, y_test)

    print("=== Ensemble (Soft Voting) ===")
    train_ensemble(rf, xgb, lgbm, X_train_res, X_test, y_train_res, y_test)

    print("Done. Run `mlflow ui` to view all experiments.")


if __name__ == "__main__":
    main()
