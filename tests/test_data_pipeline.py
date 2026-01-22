# tests/test_data_pipeline.py

def test_target_is_binary(sample_df):
    assert set(sample_df["Churn"].unique()).issubset({0, 1})

def test_required_columns_exist(sample_df):
    required_cols = {
        "customerID",
        "Churn",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    }
    assert required_cols.issubset(sample_df.columns)
