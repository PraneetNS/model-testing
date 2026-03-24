import numpy as np
import pandas as pd
import onnxruntime as ort

class ONNXModelWrapper:
    """Wrapper to make ONNX models look like Scikit-Learn estimators."""
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, X):
        if hasattr(X, "values"):
            X = X.values
        # Ensure float32 for ONNX
        X = X.astype(np.float32)
        results = self.session.run(None, {self.input_name: X})
        # If it's a classifier from skl2onnx, results[0] is labels, results[1] is probabilities
        preds = results[0]
        if isinstance(preds, (list, np.ndarray)):
            preds = np.array(preds)
        return preds

    def predict_proba(self, X):
        if hasattr(X, "values"):
            X = X.values
        X = X.astype(np.float32)
        results = self.session.run(None, {self.input_name: X})
        if len(results) > 1:
            probs = results[1]
            if isinstance(probs, list): # List of dicts case (ZipMap)
                return pd.DataFrame(probs).values
            return np.array(probs)
        return None

    @property
    def feature_names_in_(self):
        # We don't easily know this from ONNX without extra metadata
        return None
