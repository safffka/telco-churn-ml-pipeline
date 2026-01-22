# scripts/make_reference.py
import pandas as pd
from catboost import CatBoostClassifier

df = pd.read_parquet("data/processed/features.parquet").sample(300, random_state=1)

model = CatBoostClassifier()
model.load_model("models/model.cbm")

X = df.drop(columns=["Churn", "customerID"])
df["prediction"] = model.predict_proba(X)[:, 1]

df.to_parquet("data/monitoring/reference.parquet", index=False)

print("Saved reference with prediction:", df.shape)
