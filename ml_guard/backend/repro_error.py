import pandas as pd
import numpy as np
import joblib

# 1. Load model
model = joblib.load('fair_loan_model.pkl')
print(f"Model expects: {model.n_features_in_}")

# 2. Load data
df = pd.read_csv('fair_loan_test.csv')
print(f"CSV Columns: {df.columns.tolist()}")

# 3. Simulate fairness.py logic
label_col = "loan_approved"
sensitive_column = "gender"

feature_cols = [c for c in df.columns if c not in [label_col, sensitive_column]]
print(f"Feature cols identified: {feature_cols}")

X = df[feature_cols].select_dtypes(include=[np.number])
print(f"Numeric features for prediction: {X.columns.tolist()}")
print(f"Shape: {X.shape}")

try:
    preds = model.predict(X.values)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
