# tests/conftest.py
import pandas as pd
import pytest

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customerID": ["1", "2", "3", "4"],
        "Churn": [1, 0, 0, 1],
        "tenure": [1, 12, 5, 24],
        "MonthlyCharges": [70.0, 50.0, 40.0, 80.0],
        "TotalCharges": [70.0, 600.0, 200.0, 1920.0],
        "gender": ["Male", "Female", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0],
        "Partner": ["Yes", "No", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No"],
        "PhoneService": ["Yes", "Yes", "Yes", "Yes"],
        "MultipleLines": ["No", "Yes", "No", "Yes"],
        "InternetService": ["Fiber optic", "DSL", "DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes", "No", "Yes"],
        "OnlineBackup": ["Yes", "No", "Yes", "No"],
        "DeviceProtection": ["No", "Yes", "No", "Yes"],
        "TechSupport": ["No", "Yes", "No", "Yes"],
        "StreamingTV": ["Yes", "No", "Yes", "No"],
        "StreamingMovies": ["No", "Yes", "No", "Yes"],
        "Contract": ["Month-to-month", "Two year", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check",
                          "Bank transfer (automatic)", "Credit card (automatic)"],
        "is_long_contract": [0, 1, 0, 1],
        "has_fiber": [1, 0, 0, 1],
    })
