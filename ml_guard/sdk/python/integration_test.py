import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml_guard import Guard
import os
import sys

def run_integration_test():
    """
    Verifies end-to-end communication between the SDK/CLI and the Backend.
    """
    print("🚀 Starting ML Guard SDK Integration Test...")
    
    # 1. Create a dummy model and data
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    df["target"] = y
    
    model = RandomForestClassifier()
    model.fit(X, y)
    
    train_df = df.sample(frac=0.8)
    val_df = df.drop(train_df.index)
    
    # 2. Initialize Guard
    # Note: Ensure the backend is running at this URL or set via env
    backend_url = os.getenv("MLGUARD_URL", "http://localhost:8001/api/v1")
    guard = Guard(project="integration-test-project", api_url=backend_url)
    
    try:
        print(f"📡 Sending evaluation request to {backend_url}...")
        results = guard.evaluate(
            model=model,
            train_df=train_df,
            val_df=val_df,
            target_column="target",
            query="accuracy and fairness scan"
        )
        
        print("\n✅ Integration Test Passed!")
        print(f"Run ID: {results.get('run_id')}")
        print(f"Score:  {results.get('score')}")
        print(f"Risk:   {results.get('risk_level')}")
        
    except Exception as e:
        print(f"\n❌ Integration Test Failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_integration_test()
