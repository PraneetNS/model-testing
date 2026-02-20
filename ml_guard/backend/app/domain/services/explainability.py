import shap
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import structlog
from typing import Any, Dict

logger = structlog.get_logger(__name__)

class ExplainabilityService:
    @staticmethod
    def calculate_shap_values(model: Any, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate SHAP values for a model and return importance metrics.
        """
        try:
            # SHAP works differently for different model types
            explainer = None
            if "forest" in str(type(model)).lower() or "tree" in str(type(model)).lower():
                explainer = shap.TreeExplainer(model)
            else:
                # KernelExplainer is slow, but works for everything. 
                # For production, we sample the background data.
                background = shap.sample(X, 10)
                explainer = shap.KernelExplainer(model.predict, background)
            
            shap_values = explainer.shap_values(X)
            
            # For classification, shap_values might be a list (one per class)
            if isinstance(shap_values, list):
                # Typically take class 1 (churn) for binary
                importance = np.abs(shap_values[1]).mean(0)
            else:
                importance = np.abs(shap_values).mean(0)
            
            # Map features to importance
            features = X.columns.tolist()
            feature_importance = {f: float(i) for f, i in zip(features, importance)}
            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
            
            return {
                "global_importance": sorted_importance,
                "status": "success"
            }
        except Exception as e:
            logger.error("SHAP calculation failed", error=str(e))
            return {"status": "error", "message": str(e)}

    @staticmethod
    def get_shap_plot_base64(model: Any, X: pd.DataFrame):
        """Generates a summary plot as a base64 image."""
        try:
            plt.figure()
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            shap.summary_plot(shap_values, X, show=False)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        except:
            return None
