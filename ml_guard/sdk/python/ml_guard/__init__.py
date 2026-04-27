"""
ml_guard — Python SDK for ML Guard v7.2

The ML governance and observability platform that goes beyond
Evidently AI and WhyLabs.

Quick Start:
    import ml_guard

    # Initialize client
    client = ml_guard.Client(
        host="http://localhost:8000",
        api_key="mlg_xxx"
    )

    # Log a prediction (fire-and-forget)
    client.log("churn-v2", features={"age": 34}, prediction=1, proba=0.87)

    # Profile a dataset
    profile = ml_guard.profile.from_dataframe(df, "churn-v2", client=client)
    profile.flush()

    # Wrap a sklearn model (zero-code instrumentation)
    monitored = ml_guard.wrap_sklearn(model, model_id="churn-v2", client=client)
    predictions = monitored.predict(X)  # auto-logged

    # Test Suite (Evidently-style but governance-aware)
    suite = ml_guard.Suite("churn-v2", "Production Gate")
    suite.add(ml_guard.tests.accuracy_above(0.85))
    suite.add(ml_guard.tests.drift_psi_below(0.25))
    results = suite.run(df_reference=train_df, df_current=prod_df, model=model)
    results.print_summary()

    # Decorators
    @ml_guard.monitor(model_id="churn-v2")
    def predict(features: dict) -> float:
        return model.predict_proba([list(features.values())])[0][1]

    @ml_guard.gate(model_id="churn-v2", min_score=80.0)
    def deploy_to_prod():
        ...
"""
from .client import MLGuardClient, Guard, GuardrailBlockedError

from . import profile
from .suite import Suite, tests
from .decorators import monitor, gate, profile_input, trace_prediction
from .integrations import wrap_sklearn, wrap_xgboost
from .logger import PredictionLogger

__version__ = "7.2.0"

# Convenience alias
Client = MLGuardClient

__all__ = [
    # Core client
    "Client",
    "MLGuardClient",
    "Guard",           # backward-compat alias
    # Profile module
    "profile",
    # Test suites
    "Suite",
    "tests",
    # Decorators
    "monitor",
    "gate",
    "profile_input",
    "trace_prediction",
    # Integrations
    "wrap_sklearn",
    "wrap_xgboost",
    # Utilities
    "PredictionLogger",
    "GuardrailBlockedError",
]

