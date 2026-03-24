
import joblib
import pandas as pd

try:
    model = joblib.load("fair_loan_model.pkl")
    print("Model loaded successfully.")
    
    if isinstance(model, dict):
        print("Model is a dict with keys:", model.keys())
        est = model.get("model", model.get("classifier", list(model.values())[0]))
    else:
        est = model
        
except Exception as e:
    print("Error:", e)
