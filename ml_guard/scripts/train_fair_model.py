import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def generate_fair_loan_data(n_samples=2000):
    np.random.seed(42)
    
    # Features
    annual_income = np.random.normal(60000, 15000, n_samples)
    credit_score = np.random.normal(700, 50, n_samples).astype(int)
    loan_amount = np.random.normal(200000, 50000, n_samples)
    years_at_job = np.random.normal(5, 3, n_samples).clip(0, 40)
    debt_ratio = np.random.uniform(0.1, 0.5, n_samples)
    
    # Sensitive feature: Gender (balanced)
    # We consciously make sure it's not a predictor of our target
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.5, 0.5])
    
    # Target: Loan Approved (1 or 0)
    # Decisions based purely on financials, NOT on gender
    # Approval logic: (credit_score > 680 and annual_income > 45000) or (credit_score > 740)
    def approval_logic(income, credit, debt):
        # Base probability from score and income
        prob = 0.0
        if credit > 720: prob += 0.6
        elif credit > 650: prob += 0.3
        
        if income > 80000: prob += 0.3
        elif income > 50000: prob += 0.1
        
        if debt < 0.25: prob += 0.1
        
        # Add a bit of noise (non-deterministic real-world effect)
        final_prob = np.clip(prob + np.random.uniform(-0.1, 0.1), 0, 1)
        return 1 if final_prob > 0.5 else 0

    target = [approval_logic(i, c, d) for i, c, d in zip(annual_income, credit_score, debt_ratio)]
    
    # Create DataFrame
    df = pd.DataFrame({
        "annual_income": annual_income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "years_at_job": years_at_job,
        "debt_ratio": debt_ratio,
        "gender": gender,
        "target": target
    })
    
    return df

print("Generating high-quality loan dataset...")
df = generate_fair_loan_data(3000)

# 1. Save data for uploading to the portal
# 'target' is the default for UI, so we use it here.
X_data = df.drop(columns=["target"])
y_data = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Save CSVs
train_csv = "fair_loan_train.csv"
test_csv = "fair_loan_test.csv"
pd.concat([X_train, y_train], axis=1).to_csv(train_csv, index=False)
pd.concat([X_test, y_test], axis=1).to_csv(test_csv, index=False)
print(f"Saved: {train_csv}, {test_csv}")

# 2. Train Model
# Important: We must not give 'gender' to the model to ensure it's fair by blindness,
# and we'll check its performance on the Fairness tab.
print("Training model (RandomForestClassifier)...")
X_train_numeric = X_train.select_dtypes(include=[np.number])
X_test_numeric = X_test.select_dtypes(include=[np.number])

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train_numeric, y_train)

# Accuracy check
score = model.score(X_test_numeric, y_test)
print(f"Model Accuracy: {score:.4f}")

# Save Model
model_file = "fair_loan_model.pkl"
joblib.dump(model, model_file)
print(f"Saved: {model_file}")

print("\n--- Summary ---")
print(f"Label Column: target")
print(f"Sensitive Column: gender")
print(f"Model File: {os.path.abspath(model_file)}")
print(f"Data File: {os.path.abspath(test_csv)}")
print(f"Accuracy: {score*100:.2f}%")
