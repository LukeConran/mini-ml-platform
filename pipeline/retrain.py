import argparse
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).parent.parent
mlflow.set_tracking_uri(f"sqlite:///{ROOT}/mlflow.db")
import pandas as pd
from imblearn.over_sampling import SMOTE

from dataset import preprocess, PREPROCESSED_PATH
from train import load_data, train_random_forest, train_xgboost, train_lightgbm, train_ensemble

RANDOM_STATE = 42
MODEL_NAME = "churn-model"


def get_production_recall():
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "production")
        return client.get_run(mv.run_id).data.metrics.get("recall_churn", 0.0)
    except Exception:
        return 0.0  # no production model yet


def merge_new_data(new_data_path: str):
    """Preprocess new raw data, append to preprocessed.csv, and save."""
    new_df = preprocess(pd.read_csv(new_data_path))
    existing_df = pd.read_csv(PREPROCESSED_PATH)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.to_csv(PREPROCESSED_PATH, index=False)
    print(f"Merged {len(new_df)} new rows into {PREPROCESSED_PATH} ({len(combined)} total)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-data", help="Path to a raw CSV of new labeled data to incorporate before retraining")
    args = parser.parse_args()

    if args.new_data:
        merge_new_data(args.new_data)

    mlflow.set_experiment("churn-prediction")

    X_train, X_test, y_train, y_test = load_data()
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    rf   = train_random_forest(X_train_res, X_test, y_train_res, y_test)
    xgb  = train_xgboost(X_train_res, X_test, y_train_res, y_test)
    lgbm = train_lightgbm(X_train_res, X_test, y_train_res, y_test)
    train_ensemble(rf, xgb, lgbm, X_train_res, X_test, y_train_res, y_test)

    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name("churn-prediction")
    runs = client.search_runs(
        experiment.experiment_id,
        order_by=["metrics.recall_churn DESC"],
        max_results=1,
    )
    best_run = runs[0]
    new_recall = best_run.data.metrics.get("recall_churn", 0.0)
    prod_recall = get_production_recall()

    print(f"\nProduction recall: {prod_recall:.4f} | New best recall: {new_recall:.4f}")

    if new_recall > prod_recall:
        print("New model is better — promoting to Production.")
        mv = mlflow.register_model(f"runs:/{best_run.info.run_id}/model", MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, "production", mv.version)
        print(f"Registered {MODEL_NAME} v{mv.version} as @production.")
    else:
        print("Existing production model is still the best. No update made.")


if __name__ == "__main__":
    main()
