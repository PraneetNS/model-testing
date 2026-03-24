import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

# Create samples directory
os.makedirs("samples", exist_ok=True)

# 1. Create a dummy model
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)
model = RandomForestClassifier()
model.fit(X, y)

with open("samples/sample_model.pkl", "wb") as f:
    pickle.dump(model, f)

# 2. Create sample training data
train_df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3", "feature4"])
train_df["target"] = y
train_df.to_csv("samples/sample_train.csv", index=False)

# 3. Create sample validation data (with slight drift)
X_val = np.random.rand(50, 4) + 0.1  # Add slight shift for drift detection
y_val = np.random.randint(0, 2, 50)
val_df = pd.DataFrame(X_val, columns=["feature1", "feature2", "feature3", "feature4"])
val_df["target"] = y_val
val_df.to_csv("samples/sample_val.csv", index=False)

print("Created sample files in /samples directory:")
print("- samples/sample_model.pkl")
print("- samples/sample_train.csv")
print("- samples/sample_val.csv")
