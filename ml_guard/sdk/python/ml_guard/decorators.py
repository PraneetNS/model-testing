"""
ml_guard/decorators.py — Instrumentation Decorators

ML Guard's unique differentiator over Evidently and WhyLabs:
Zero-boilerplate, decorator-based instrumentation that auto-captures:
  - Prediction inputs/outputs
  - Latency (wall + CPU time)
  - Exceptions and error rates
  - Input drift warnings

Usage:
    from ml_guard.decorators import monitor, profile_input, gate

    @monitor(model_id="churn-v2")
    def predict(features: dict) -> float:
        return model.predict_proba([list(features.values())])[0][1]

    @profile_input(model_id="churn-v2", every_n=100)
    def predict_batch(df: pd.DataFrame):
        return model.predict(df)

    @gate(model_id="churn-v2", min_score=75.0)
    def deploy_model():
        # blocked unless governance score >= 75
        ...
"""
from __future__ import annotations

import functools
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger("mlguard.decorators")

F = TypeVar("F", bound=Callable[..., Any])

# Thread-local counter for batch profiling
_call_counters: Dict[str, int] = {}
_counter_lock = threading.Lock()


def _get_client():
    """Lazily import and return the default MLGuardClient from env vars."""
    try:
        from ml_guard.client import MLGuardClient
        return MLGuardClient(
            host=os.getenv("MLGUARD_HOST", "http://localhost:8000"),
            api_key=os.getenv("MLGUARD_API_KEY", ""),
        )
    except Exception as e:
        logger.warning(f"mlguard_client_init_failed: {e}")
        return None


# ── @monitor ────────────────────────────────────────────────────────────────────

def monitor(
    model_id: str,
    environment: str = "production",
    log_inputs: bool = True,
    log_outputs: bool = True,
    client=None,
    async_mode: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to automatically log every prediction to ML Guard.
    Captures features, prediction, latency, and errors.

    Example:
        @monitor(model_id="fraud-detector-v3")
        def predict(features: dict) -> float:
            return model.predict_proba([list(features.values())])[0][1]

        result = predict({"age": 34, "spend": 800})
        # → automatically logged to ML Guard
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _client = client or _get_client()
            start = time.perf_counter()
            exc_occurred = False
            result = None

            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as e:
                exc_occurred = True
                raise
            finally:
                latency_ms = (time.perf_counter() - start) * 1000

                if _client is not None:
                    # Extract features: prefer first dict arg or kwargs
                    features = {}
                    if log_inputs:
                        for arg in args:
                            if isinstance(arg, dict):
                                features = arg
                                break
                        if not features:
                            features = {k: v for k, v in kwargs.items()
                                       if not callable(v)}

                    prediction_val = None
                    if log_outputs and result is not None and not exc_occurred:
                        if isinstance(result, (int, float, str, bool)):
                            prediction_val = result
                        elif isinstance(result, (list, tuple)) and len(result) > 0:
                            prediction_val = result[0]

                    def _send():
                        try:
                            _client.log(
                                model_id=model_id,
                                features=features,
                                prediction=str(prediction_val) if prediction_val is not None else None,
                                latency_ms=latency_ms,
                                environment=environment,
                                tags={
                                    "fn": fn.__name__,
                                    "error": exc_occurred,
                                    "decorator": "monitor",
                                },
                            )
                        except Exception as log_err:
                            logger.debug(f"monitor_log_failed: {log_err}")

                    if async_mode:
                        threading.Thread(target=_send, daemon=True).start()
                    else:
                        _send()

        return wrapper  # type: ignore
    return decorator


# ── @profile_input ───────────────────────────────────────────────────────────

def profile_input(
    model_id: str,
    every_n: int = 100,
    dataset_name: str = "production",
    client=None,
) -> Callable[[F], F]:
    """
    Decorator that profiles DataFrame inputs every N calls and uploads
    the lightweight profile to ML Guard (no raw data transmitted).

    Best for batch predict functions:

        @profile_input(model_id="churn-v2", every_n=500)
        def predict_batch(df: pd.DataFrame):
            return model.predict(df)
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Find the DataFrame argument
            import pandas as pd
            df_arg = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df_arg = arg
                    break
            if df_arg is None:
                for v in kwargs.values():
                    if isinstance(v, pd.DataFrame):
                        df_arg = v
                        break

            result = fn(*args, **kwargs)

            if df_arg is not None:
                with _counter_lock:
                    count = _call_counters.get(model_id, 0) + 1
                    _call_counters[model_id] = count

                if count % every_n == 0:
                    _client = client or _get_client()
                    if _client is not None:
                        def _send_profile():
                            try:
                                from ml_guard.profile import from_dataframe
                                prof = from_dataframe(
                                    df_arg, model_id=model_id,
                                    dataset_name=dataset_name,
                                    client=_client,
                                )
                                _client.upload_profile(prof)
                                logger.info(
                                    f"profile_uploaded model_id={model_id} "
                                    f"call_count={count} rows={len(df_arg)}"
                                )
                            except Exception as e:
                                logger.debug(f"profile_upload_failed: {e}")

                        threading.Thread(target=_send_profile, daemon=True).start()

            return result
        return wrapper  # type: ignore
    return decorator


# ── @gate ────────────────────────────────────────────────────────────────────

def gate(
    model_id: str,
    min_score: float = 70.0,
    policy_config: Optional[Dict[str, Any]] = None,
    client=None,
    fail_open: bool = False,
) -> Callable[[F], F]:
    """
    Decorator that blocks function execution unless the model's
    governance score meets the required threshold.

    Perfect for deploy() functions in CI:

        @gate(model_id="fraud-v3", min_score=80.0)
        def deploy_to_production():
            kubernetes.deploy(...)

    Args:
        model_id: Model to check governance score for
        min_score: Minimum governance score to allow execution
        policy_config: Optional policy override dict
        fail_open: If True, allow execution when ML Guard is unreachable
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _client = client or _get_client()

            if _client is None:
                if fail_open:
                    logger.warning(f"gate_skipped model_id={model_id} reason=no_client")
                    return fn(*args, **kwargs)
                raise RuntimeError(
                    f"[ml_guard] Governance gate: cannot connect to ML Guard backend. "
                    f"Set MLGUARD_HOST and MLGUARD_API_KEY or use fail_open=True."
                )

            try:
                score_data = _client.get_score(model_id)
                actual_score = score_data.get("overall_score", 0)
                verdict = score_data.get("verdict", "UNKNOWN")

                if actual_score < min_score:
                    raise RuntimeError(
                        f"[ml_guard] @gate BLOCKED: model '{model_id}' governance score "
                        f"{actual_score:.1f} < required {min_score:.1f} "
                        f"(verdict={verdict}). Run an audit to improve your score."
                    )

                logger.info(
                    f"gate_passed model_id={model_id} "
                    f"score={actual_score:.1f} threshold={min_score}"
                )
                return fn(*args, **kwargs)

            except RuntimeError:
                raise
            except Exception as e:
                if fail_open:
                    logger.warning(f"gate_error_fail_open model_id={model_id} error={e}")
                    return fn(*args, **kwargs)
                raise RuntimeError(
                    f"[ml_guard] Governance gate check failed: {e}"
                ) from e

        return wrapper  # type: ignore
    return decorator


# ── @trace_prediction ────────────────────────────────────────────────────────

def trace_prediction(
    model_id: str,
    feature_extractor: Optional[Callable] = None,
    output_extractor: Optional[Callable] = None,
    client=None,
) -> Callable[[F], F]:
    """
    Advanced decorator with custom feature/output extraction logic.
    Use when predict() doesn't take a plain dict as first arg.

    Example:
        def extract_features(raw_request) -> dict:
            return {\"age\": raw_request.user.age, \"plan\": raw_request.plan}

        @trace_prediction(
            model_id=\"churn-v2\",
            feature_extractor=extract_features,
        )
        def predict(raw_request: MyRequest):
            return model.predict(...)
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            _client = client or _get_client()
            if _client is not None:
                features = {}
                if feature_extractor:
                    try:
                        features = feature_extractor(*args, **kwargs) or {}
                    except Exception:
                        pass

                prediction = None
                if output_extractor:
                    try:
                        prediction = output_extractor(result)
                    except Exception:
                        pass
                elif isinstance(result, (int, float, str, bool)):
                    prediction = result

                def _send():
                    try:
                        _client.log(
                            model_id=model_id,
                            features=features,
                            prediction=str(prediction) if prediction is not None else None,
                            latency_ms=latency_ms,
                            tags={"fn": fn.__name__, "decorator": "trace_prediction"},
                        )
                    except Exception:
                        pass

                threading.Thread(target=_send, daemon=True).start()

            return result
        return wrapper  # type: ignore
    return decorator
