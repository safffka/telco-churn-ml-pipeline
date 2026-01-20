# src/data_pipeline.py
import pandas as pd
from typing import Tuple, List

TARGET_COL = "Churn"
ID_COL = "customerID"

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # target to binary
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

    # numeric coercion
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # drop rows with missing values
    df = df.dropna().reset_index(drop=True)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # business features
    df["avg_charge_per_month"] = (
        df["TotalCharges"] / (df["tenure"] + 1)
    )

    df["is_long_contract"] = df["Contract"].isin(
        ["One year", "Two year"]
    ).astype(int)

    df["has_fiber"] = (
        df["InternetService"] == "Fiber optic"
    ).astype(int)

    return df


def get_feature_lists() -> Tuple[List[str], List[str]]:
    categorical = CATEGORICAL_COLS + [
        "is_long_contract",
        "has_fiber",
    ]
    numeric = NUMERIC_COLS + ["avg_charge_per_month"]

    return categorical, numeric


def save_features(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)


def run_pipeline(
    input_path: str = "data/raw/telco.csv",
    output_path: str = "data/processed/features.parquet",
) -> None:
    df = load_data(input_path)
    df = clean_data(df)
    df = build_features(df)
    save_features(df, output_path)


if __name__ == "__main__":
    run_pipeline()
