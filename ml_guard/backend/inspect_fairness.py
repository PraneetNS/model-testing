import joblib
import pandas as pd
m = joblib.load('fair_loan_model.pkl')
print(f"Features: {m.n_features_in_}")
df = pd.read_csv('fair_loan_test.csv')
print(f"Columns: {df.columns.tolist()}")
numeric = df.select_dtypes(include=['number']).columns.tolist()
print(f"Numeric: {numeric}")
