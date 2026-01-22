import pandas as pd
from catboost import CatBoostClassifier

df = pd.read_parquet("data/processed/features.parquet").sample(300, random_state=7)

model = CatBoostClassifier()
model.load_model("models/model.cbm")

X = df.drop(columns=["Churn", "customerID"])
df["prediction"] = model.predict_proba(X)[:, 1]

df.to_parquet("data/monitoring/current.parquet", index=False)

print("Saved: data/monitoring/current.parquet", df.shape)
