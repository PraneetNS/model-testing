import pandas as pd
import numpy as np
from niyantrana.local.explainability import explain

class MockModel:
    def predict(self, X):
        return np.sum(X.values, axis=1)
        
    def predict_proba(self, X):
        # Dummy proba
        return np.array([[0.5, 0.5] for _ in range(len(X))])

def test_explain():
    # We will only test the mock or the exception if shap/lime are not installed.
    # To keep tests robust without requiring heavy dependencies for basic tests,
    # we'll just check that it raises ImportError if shap isn't installed, 
    # or runs if it is.
    df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    model = MockModel()
    
    try:
        report = explain(model, df, method="shap")
        assert report.method == "shap"
        assert len(report.feature_importances) == 2
        assert report.feature_importances[0].rank == 1
    except ImportError:
        pass # Expected if shap is not installed in the test environment

    try:
        report = explain(model, df, method="lime")
        assert report.method == "lime"
        assert len(report.feature_importances) == 2
    except ImportError:
        pass
