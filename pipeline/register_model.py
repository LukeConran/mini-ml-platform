"""
Manual model promotion tool.

Usage:
  python register_model.py <run_id>                    # register and set @production alias
  python register_model.py <run_id> --alias staging
"""
import argparse
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).parent.parent
mlflow.set_tracking_uri(f"sqlite:///{ROOT}/mlflow.db")

MODEL_NAME = "churn-model"


def main():
    parser = argparse.ArgumentParser(description="Register an MLflow run as a named model version.")
    parser.add_argument("run_id", help="MLflow run ID to register")
    parser.add_argument("--alias", default="production", help="Alias to assign (default: production)")
    args = parser.parse_args()

    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{args.run_id}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, args.alias, mv.version)
    print(f"Registered {MODEL_NAME} v{mv.version} → @{args.alias}")


if __name__ == "__main__":
    main()
