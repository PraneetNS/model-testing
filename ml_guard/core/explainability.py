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

def run_explainability(model, X, feature_names, max_samples=100):
    """Entry point for the explainability task."""
    import shap
    
    # 1. Sample Down
    if len(X) > max_samples:
        X_sample = X[np.random.choice(len(X), max_samples, replace=False)]
    else:
        X_sample = X
        
    X_df = pd.DataFrame(X_sample, columns=feature_names)
    
    # 2. Compute SHAP
    explainer = shap.Explainer(model, X_df)
    shap_values = explainer(X_df)
    
    # 3. Process Importances
    vals = shap_values.values
    if len(vals.shape) > 2:
        vals = np.abs(vals).mean(axis=2)
    
    mean_abs_shap = np.abs(vals).mean(axis=0)
    
    feat_imp = {}
    total_imp = np.sum(mean_abs_shap)
    for i, feat in enumerate(feature_names):
        feat_imp[feat] = float(mean_abs_shap[i] / total_imp) if total_imp > 0 else 0
        
    # 4. Interpretability Score (Feature Concentration)
    # How much of the model's behavior is driven by the top 3 features?
    sorted_imps = sorted(feat_imp.values(), reverse=True)
    top_3_sum = sum(sorted_imps[:3]) if len(sorted_imps) >= 3 else sum(sorted_imps)
    
    if total_imp == 0:
        # Constant model is highly interpretable (interp_score = 100)
        interp_score = 100.0
    else:
        # We use a Gini-inspired concentration score
        # If top features dominate, it's highly interpretable.
        interp_score = min(100.0, top_3_sum * 100.0)
        
    # Adjustment for very small feature sets
    if len(feature_names) < 3:
        interp_score = 100.0

    return {
        "method": "shap",
        "feature_importance": feat_imp,
        "interpretability_score": interp_score,
        "top_features": sorted(feat_imp.keys(), key=lambda x: feat_imp[x], reverse=True)[:5]
    }
