"""
ml_guard/integrations/__init__.py — ML Framework Integrations

ML Guard integrations for sklearn, XGBoost, and HuggingFace.
Zero-code wrappers that auto-instrument existing model objects:

    from ml_guard.integrations import sklearn, xgboost, huggingface

    # Wrap any sklearn pipeline
    monitored_model = sklearn.wrap(model, model_id="churn-v2", client=client)
    monitored_model.predict(X)  # auto-logged
"""
from .sklearn_integration import wrap_sklearn
from .xgboost_integration import wrap_xgboost

__all__ = ["wrap_sklearn", "wrap_xgboost"]
