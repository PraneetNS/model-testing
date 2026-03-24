
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_feature_importance(model, X, feature_names=None):
    """Compute feature importance from a trained model."""
    importance = {}
    try:
        # Tree-based models have feature_importances_
        if hasattr(model, "feature_importances_"):
            raw = model.feature_importances_
            names = feature_names or [f"feature_{i}" for i in range(len(raw))]
            importance = {name: float(val) for name, val in zip(names, raw)}
        # Linear models have coef_
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_)
            if coef.ndim > 1:
                coef = coef.mean(axis=0)
            names = feature_names or [f"feature_{i}" for i in range(len(coef))]
            total = coef.sum() if coef.sum() > 0 else 1
            importance = {name: float(val / total) for name, val in zip(names, coef)}
        else:
            # Permutation-based fallback
            importance = _permutation_importance(model, X, feature_names)
    except Exception as e:
        logger.warning("Feature importance computation failed: %s", e)
        importance = {}
    
    # Sort by importance descending
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def compute_shap_global(model, X, feature_names=None, max_samples=100):
    """Compute SHAP global feature explanations (sampled for performance)."""
    try:
        import shap
        
        X_sample = X[:max_samples] if len(X) > max_samples else X
        names = feature_names or [f"feature_{i}" for i in range(X_sample.shape[1])]
        
        # Use TreeExplainer for tree models, KernelExplainer as fallback
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, min(10, len(X_sample))))
        
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        global_importance = {name: float(val) for name, val in zip(names, mean_abs_shap)}
        
        return dict(sorted(global_importance.items(), key=lambda x: x[1], reverse=True))
    except ImportError:
        logger.info("SHAP not installed; falling back to feature_importance.")
        return compute_feature_importance(model, X, feature_names)
    except Exception as e:
        logger.warning("SHAP computation failed: %s. Falling back.", e)
        return compute_feature_importance(model, X, feature_names)


def compute_shap_local(model, X, sample_indices=None, feature_names=None, max_samples=5):
    """Compute SHAP local explanations for specific samples."""
    try:
        import shap
        
        if sample_indices is None:
            sample_indices = list(range(min(max_samples, len(X))))
        
        X_explain = X[sample_indices] if hasattr(X, '__getitem__') else X.iloc[sample_indices]
        names = feature_names or [f"feature_{i}" for i in range(X_explain.shape[1])]
        
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            bg = shap.sample(X, min(10, len(X)))
            explainer = shap.KernelExplainer(model.predict, bg)
        
        shap_values = explainer.shap_values(X_explain)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        explanations = []
        for i, idx in enumerate(sample_indices):
            row_exp = {name: float(shap_values[i][j]) for j, name in enumerate(names)}
            explanations.append({
                "sample_index": int(idx),
                "shap_values": dict(sorted(row_exp.items(), key=lambda x: abs(x[1]), reverse=True)),
            })
        
        return explanations
    except ImportError:
        return [{"error": "SHAP library not installed"}]
    except Exception as e:
        return [{"error": str(e)}]


def compute_interpretability_score(feature_importance, model=None):
    """Compute an interpretability score (0-100) based on feature concentration."""
    if not feature_importance:
        return 50.0
    
    values = list(feature_importance.values())
    total = sum(values) if sum(values) > 0 else 1
    normalized = [v / total for v in values]
    
    # Calculate entropy-based interpretability
    entropy = -sum(p * np.log2(p + 1e-10) for p in normalized if p > 0)
    max_entropy = np.log2(max(len(normalized), 1))
    
    # Lower entropy = more concentrated = more interpretable
    concentration = 1 - (entropy / max_entropy) if max_entropy > 0 else 0.5
    
    # Factor in the number of features (fewer features = more interpretable)
    feature_penalty = min(1.0, 10 / max(len(normalized), 1))
    
    score = (concentration * 70 + feature_penalty * 30)
    return round(min(100, max(0, score)), 1)


def run_explainability(model, X, feature_names=None, max_samples=100):
    """Full explainability pipeline: importance + SHAP global + interpretability score."""
    importance = compute_feature_importance(model, X, feature_names)
    shap_global = compute_shap_global(model, X, feature_names, max_samples)
    interpretability = compute_interpretability_score(shap_global or importance, model)
    
    return {
        "feature_importance": importance,
        "shap_global": shap_global,
        "interpretability_score": interpretability,
        "top_features": list((shap_global or importance).keys())[:5],
        "method": "shap" if shap_global != importance else "native",
    }


def _permutation_importance(model, X, feature_names=None, n_repeats=5):
    """Simple permutation importance as a fallback."""
    try:
        from sklearn.inspection import permutation_importance as sklearn_pi
        result = sklearn_pi(model, X, np.zeros(len(X)), n_repeats=n_repeats, random_state=42)
        names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        return {name: float(val) for name, val in zip(names, result.importances_mean)}
    except Exception:
        return {}
