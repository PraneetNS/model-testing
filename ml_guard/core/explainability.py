import numpy as np
import pandas as pd
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

def generate_shap_explanation(model, X_reference: pd.DataFrame, X_current: pd.DataFrame = None):
    """
    Computes SHAP-based feature importance and drift delta vs baseline.
    Returns:
        {
            "feature_importances": [
                {"feature": str, "mean_abs_shap": float, "rank": int}
            ],
            "top_drift_contributors": [
                {"feature": str, "shap_delta_vs_baseline": float}
            ],
            "explanation_timestamp": str
        }
    """
    import shap
    
    # Auto-select explainer based on model
    explainer = shap.Explainer(model, X_reference)

    # Compute SHAP on reference data
    shap_vals_ref = explainer(X_reference)
    
    # shap_vals_ref.values may be 3D for multi-class. Condense to 2D by taking mean across classes if needed.
    vals_ref = shap_vals_ref.values
    if len(vals_ref.shape) > 2:
        vals_ref = np.abs(vals_ref).mean(axis=2)
    
    mean_abs_ref = np.abs(vals_ref).mean(axis=0)

    features = list(X_reference.columns)
    
    # Create feature importances list
    importances = []
    for i, feature in enumerate(features):
        importances.append({
            "feature": feature,
            "mean_abs_shap": float(mean_abs_ref[i])
        })
        
    # Sort and add ranks
    importances.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
    for rank, item in enumerate(importances, start=1):
        item["rank"] = rank

    top_drift_contributors = []

    # If current data is provided, calculate drift
    if X_current is not None:
        shap_vals_curr = explainer(X_current)
        vals_curr = shap_vals_curr.values
        if len(vals_curr.shape) > 2:
            vals_curr = np.abs(vals_curr).mean(axis=2)
        
        mean_abs_curr = np.abs(vals_curr).mean(axis=0)
        
        drift_deltas = []
        for i, feature in enumerate(features):
            delta = float(mean_abs_curr[i] - mean_abs_ref[i])
            drift_deltas.append({
                "feature": feature,
                "shap_delta_vs_baseline": delta,
                "abs_delta": abs(delta)
            })
            
        drift_deltas.sort(key=lambda x: x["abs_delta"], reverse=True)
        # Drop the abs_delta helper key
        for item in drift_deltas[:10]: # Return top 10
            top_drift_contributors.append({
                "feature": item["feature"],
                "shap_delta_vs_baseline": item["shap_delta_vs_baseline"]
            })

    return {
        "feature_importances": importances,
        "top_drift_contributors": top_drift_contributors,
        "explanation_timestamp": datetime.now(timezone.utc).isoformat()
    }
