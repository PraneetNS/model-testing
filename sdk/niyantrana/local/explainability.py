import pandas as pd
import numpy as np
from typing import Literal, Any
from ..models import ExplainReport, FeatureImportance

def explain(model: Any, X: pd.DataFrame, method: Literal["shap", "lime"] = "shap", n_samples: int = 100) -> ExplainReport:
    """
    Generates feature importances using SHAP or LIME.
    """
    feature_importances = []
    
    if method == "shap":
        try:
            import shap
            
            # Using a sample of X for explanation if it's large
            X_sample = shap.sample(X, min(n_samples, len(X)))
            
            # Try to determine the explainer type based on the model
            try:
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample)
                
                # Mean absolute SHAP values across samples
                if len(shap_values.values.shape) == 3: # Multi-class
                    mean_abs_shap = np.abs(shap_values.values).mean(axis=(0, 2))
                else:
                    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                    
            except Exception as e:
                # Fallback to KernelExplainer
                predict_fn = getattr(model, "predict_proba", getattr(model, "predict"))
                explainer = shap.KernelExplainer(predict_fn, X_sample)
                shap_values = explainer.shap_values(X_sample)
                
                if isinstance(shap_values, list): # Multi-class
                    mean_abs_shap = np.abs(shap_values[0]).mean(axis=0)
                else:
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            for i, feature in enumerate(X.columns):
                feature_importances.append({
                    "feature": feature,
                    "importance": float(mean_abs_shap[i])
                })
                
        except ImportError:
            raise ImportError("Please install shap to use the shap explainability method: pip install shap")
            
    elif method == "lime":
        try:
            import lime
            import lime.lime_tabular
            
            X_sample = X.sample(n=min(n_samples, len(X)))
            
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X.values,
                feature_names=X.columns.tolist(),
                mode='classification' # Defaulting to classification for simplicity
            )
            
            predict_fn = getattr(model, "predict_proba", getattr(model, "predict"))
            
            importance_dict = {f: 0.0 for f in X.columns}
            
            for i in range(len(X_sample)):
                exp = explainer.explain_instance(X_sample.values[i], predict_fn, num_features=len(X.columns))
                for f_idx, weight in exp.local_exp[list(exp.local_exp.keys())[0]]:
                    importance_dict[X.columns[f_idx]] += abs(weight)
                    
            for f in X.columns:
                importance_dict[f] /= len(X_sample)
                feature_importances.append({
                    "feature": f,
                    "importance": float(importance_dict[f])
                })
                
        except ImportError:
            raise ImportError("Please install lime to use the lime explainability method: pip install lime")

    # Sort by importance and assign ranks
    feature_importances.sort(key=lambda x: x["importance"], reverse=True)
    
    ranked_importances = [
        FeatureImportance(
            feature=fi["feature"], 
            importance=fi["importance"], 
            rank=idx + 1
        ) for idx, fi in enumerate(feature_importances)
    ]

    return ExplainReport(
        method=method,
        feature_importances=ranked_importances
    )
