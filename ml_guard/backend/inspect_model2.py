
import joblib
import pandas as pd

model = joblib.load("fair_loan_model.pkl")
if hasattr(model, "feature_names_in_"):
    print("Expected:", list(model.feature_names_in_))
df = pd.read_csv("fair_loan_test.csv")
print("CSV:", list(df.columns))
