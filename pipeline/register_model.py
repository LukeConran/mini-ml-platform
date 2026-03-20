"""
Manual model promotion tool.

Usage:
  python register_model.py <run_id>              # register and promote to Production
  python register_model.py <run_id> --stage Staging
"""
import argparse

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "churn-model"


def main():
    parser = argparse.ArgumentParser(description="Register an MLflow run as a named model version.")
    parser.add_argument("run_id", help="MLflow run ID to register")
    parser.add_argument("--stage", default="Production", choices=["Staging", "Production"],
                        help="Stage to promote the model to (default: Production)")
    args = parser.parse_args()

    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{args.run_id}/model", MODEL_NAME)
    client.transition_model_version_stage(MODEL_NAME, mv.version, stage=args.stage)
    print(f"Registered {MODEL_NAME} v{mv.version} → {args.stage}")


if __name__ == "__main__":
    main()
