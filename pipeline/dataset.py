import argparse

import kagglehub
import pandas as pd

PREPROCESSED_PATH = "data/preprocessed.csv"

BINARY_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
ONEHOT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df = pd.get_dummies(df, columns=ONEHOT_COLS)

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    df.columns = [
        col.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        for col in df.columns
    ]
    return df


def main():
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    df = pd.read_csv(path + "/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    preprocess(df).to_csv(PREPROCESSED_PATH, index=False)
    print(f"Saved preprocessed dataset to {PREPROCESSED_PATH}")


if __name__ == "__main__":
    main()
