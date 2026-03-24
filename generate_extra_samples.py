import pandas as pd
import numpy as np
import os

# Create samples directory
os.makedirs("samples", exist_ok=True)

# 1. Create probe data (Live Monitor)
# This should be just features, same columns as the model expects
X_probe = np.random.rand(20, 4)
probe_df = pd.DataFrame(X_probe, columns=["feature1", "feature2", "feature3", "feature4"])
probe_df.to_csv("samples/probe_data.csv", index=False)

print("Created sample files for testing:")
print("- samples/probe_data.csv (Use for Live Monitor)")
