import shap
import pandas as pd
import numpy as np
import asyncio
from typing import Dict, Any, List
import structlog

logger = structlog.get_logger(__name__)

class ExplainabilityEngine:
    """
    Provides model interpretation using SHAP values.
    Offloads heavy computation to a separate thread.
    """
    
    async def get_feature_importance(self, model: Any, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Calculates global feature importance using SHAP.
        """
        return await asyncio.to_thread(self._calculate_shap, model, X)

    def _calculate_shap(self, model: Any, X: pd.DataFrame) -> List[Dict[str, Any]]:
        try:
            # Sample data if too large for speed
            if len(X) > 100:
                X_sample = X.sample(100, random_state=42)
            else:
                X_sample = X

            # Handle different model types
            explainer = None
            if hasattr(model, "predict_proba"):
                try:
                    explainer = shap.TreeExplainer(model)
                except Exception:
                    # Fallback to KernelExplainer for non-tree models (slower)
                    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_sample, 10))
            
            if not explainer:
                return []

            shap_values = explainer.shap_values(X_sample)
            
            # For classification, shap_values might be a list (one per class)
            if isinstance(shap_values, list):
                # Use absolute mean across all classes or just class 1
                importance = np.abs(shap_values[1]).mean(0) if len(shap_values) > 1 else np.abs(shap_values[0]).mean(0)
            else:
                importance = np.abs(shap_values).mean(0)

            feature_importance = []
            for name, imp in zip(X.columns, importance):
                feature_importance.append({"feature": name, "importance": float(imp)})
            
            # Sort by importance
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
            return feature_importance[:10] # Top 10

        except Exception as e:
            logger.error("SHAP calculation failed", error=str(e))
            return []
