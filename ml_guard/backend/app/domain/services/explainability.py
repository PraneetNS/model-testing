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
            # Hard cap at 50 samples — KernelExplainer is O(n*features) and very slow
            n_samples = min(50, len(X))
            X_sample = X.sample(n_samples, random_state=42) if len(X) > n_samples else X.copy()

            # --- Feature alignment to match model's expectations ---
            if getattr(model, "feature_names_in_", None) is not None:
                expected = list(model.feature_names_in_)
                for f in expected:
                    if f not in X_sample.columns:
                        X_sample[f] = 0
                X_sample = X_sample[expected]
            elif getattr(model, "n_features_in_", None) is not None:
                n_expected = model.n_features_in_
                if X_sample.shape[1] < n_expected:
                    for i in range(X_sample.shape[1], n_expected):
                        X_sample[f"__pad_{i}__"] = 0
                elif X_sample.shape[1] > n_expected:
                    X_sample = X_sample.iloc[:, :n_expected]

            importance = None

            # --- Strategy 1: TreeExplainer (instant for RF, XGB, LightGBM, etc.) ---
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    importance = np.abs(shap_values[1]).mean(0) if len(shap_values) > 1 else np.abs(shap_values[0]).mean(0)
                else:
                    importance = np.abs(shap_values).mean(0)
                logger.info("SHAP: used TreeExplainer")
            except Exception:
                pass

            # --- Strategy 2: LinearExplainer (fast for linear models) ---
            if importance is None:
                try:
                    background = shap.maskers.Independent(X_sample, max_samples=25)
                    explainer = shap.LinearExplainer(model, background)
                    shap_values = explainer.shap_values(X_sample)
                    importance = np.abs(shap_values).mean(0)
                    logger.info("SHAP: used LinearExplainer")
                except Exception:
                    pass

            # --- Strategy 3: Permutation importance (no SHAP, always fast) ---
            if importance is None:
                logger.warning("SHAP explainers failed, falling back to permutation importance")
                if hasattr(model, "predict"):
                    baseline_preds = model.predict(X_sample.values)
                    importances = []
                    for i in range(X_sample.shape[1]):
                        X_perm = X_sample.values.copy()
                        np.random.shuffle(X_perm[:, i])
                        perm_preds = model.predict(X_perm)
                        importances.append(float(np.mean(np.abs(perm_preds - baseline_preds))))
                    importance = np.array(importances)
                else:
                    return []

            feature_importance = [
                {"feature": name, "importance": float(imp)}
                for name, imp in zip(X.columns, importance)
            ]
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
            return feature_importance[:10]  # Top 10

        except Exception as e:
            logger.error("SHAP calculation failed", error=str(e))
            return []
