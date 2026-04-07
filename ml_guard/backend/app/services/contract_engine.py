"""
contract_engine.py — ML Guard Model Behavior Contract Engine

Checks every incoming prediction against the model's active behavioral
contracts. Records breaches and computes governance score penalties.

Design principles:
  - Fast path: returns [] immediately when no contracts exist (0 DB reads)
  - Never raises: all exceptions are caught; ingest pipeline is safe
  - Synchronous: called inline in ingest to ensure no missed predictions
  - Target overhead: <5 ms for non-distribtuion checks (single DB query)
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class ContractEngine:
    """
    Evaluates active behavioral contracts for a model against a single
    incoming prediction. Designed to be called from the ingest pipeline
    synchronously. Fast-path exits immediately when no contracts exist.
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def check_prediction(
        self,
        db: Session,
        model_id: str,
        prediction: Any,
        prediction_proba: Optional[float],
        features: Dict[str, Any],
        latency_ms: Optional[float],
        log_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Check a single prediction against all active contracts for this model.
        """
        logger.info(f">>> check_prediction model_id={model_id} proba={prediction_proba} latency={latency_ms}")
        from app.db.models import ModelContract, ContractBreach

        # ── Fast path ─────────────────────────────────────────────────────────
        try:
            contracts = (
                db.query(ModelContract)
                .filter(
                    ModelContract.model_id == model_id,
                    ModelContract.is_active.is_(True),
                )
                .all()
            )
            logger.info(f"contract_engine found {len(contracts)} active models={model_id}")
        except Exception as e:
            logger.warning(f"contract_query_failed model_id={model_id} error={e}")
            return []

        if not contracts:
            return []

        breaches: List[Dict[str, Any]] = []

        for contract in contracts:
            promises: List[Dict[str, Any]] = contract.promises or []
            for promise in promises:
                breach = self._check_promise(
                    db=db,
                    model_id=model_id,
                    contract_id=str(contract.id),
                    promise=promise,
                    prediction=prediction,
                    prediction_proba=prediction_proba,
                    features=features,
                    latency_ms=latency_ms,
                    log_id=log_id,
                )
                if breach is None:
                    continue

                breaches.append(breach)

                # Persist the breach record (fire-and-forget safe)
                try:
                    record = ContractBreach(
                        id=uuid.uuid4(),
                        contract_id=contract.id,
                        model_id=model_id,
                        promise_name=promise.get("name", "unknown"),
                        promise_type=promise.get("type", "unknown"),
                        expected=str(promise.get("threshold", "")),
                        actual=str(breach.get("actual", "")),
                        prediction_log_id=log_id,
                        severity=promise.get("severity", "HIGH"),
                        resolved=False,
                    )
                    db.add(record)
                    db.commit()
                except Exception as e:
                    logger.warning(f"breach_persist_failed promise={promise.get('name')} error={e}")
                    try:
                        db.rollback()
                    except Exception:
                        pass

        return breaches

    def get_breach_summary(
        self,
        db: Session,
        model_id: str,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Aggregate contract breaches for a model over the last N hours.
        Used by the governance engine to compute a score penalty.

        Penalty formula (capped at 20 pts):
            CRITICAL breach  = 2.0 pts
            HIGH breach      = 1.0 pts
            MEDIUM breach    = 0.5 pts
            LOW breach       = 0.0 pts
        """
        from app.db.models import ContractBreach

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        try:
            breaches = (
                db.query(ContractBreach)
                .filter(
                    ContractBreach.model_id == model_id,
                    ContractBreach.created_at >= cutoff,
                )
                .all()
            )
        except Exception as e:
            logger.warning(f"breach_summary_failed model_id={model_id} error={e}")
            return {
                "total_breaches": 0,
                "by_severity": {},
                "by_promise": {},
                "governance_penalty": 0.0,
                "window_hours": hours,
            }

        by_severity: Dict[str, int] = {}
        by_promise: Dict[str, int] = {}

        for b in breaches:
            sev = b.severity or "HIGH"
            by_severity[sev] = by_severity.get(sev, 0) + 1
            by_promise[b.promise_name] = by_promise.get(b.promise_name, 0) + 1

        penalty = (
            by_severity.get("CRITICAL", 0) * 2.0
            + by_severity.get("HIGH", 0) * 1.0
            + by_severity.get("MEDIUM", 0) * 0.5
            + by_severity.get("LOW", 0) * 0.0
        )

        return {
            "total_breaches": len(breaches),
            "by_severity": by_severity,
            "by_promise": by_promise,
            "governance_penalty": round(min(penalty, 20.0), 2),
            "window_hours": hours,
        }

    # ── Promise evaluation ─────────────────────────────────────────────────────

    def _check_promise(
        self,
        db: Session,
        model_id: str,
        contract_id: str,
        promise: Dict[str, Any],
        prediction: Any,
        prediction_proba: Optional[float],
        features: Dict[str, Any],
        latency_ms: Optional[float],
        log_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single promise against the current prediction.
        Returns a breach dict if violated, None if compliant.
        """
        ptype: str = promise.get("type", "output")
        operator: str = promise.get("operator", "gte")
        threshold: float = float(promise.get("threshold", 0))
        metric: str = promise.get("metric", "")
        name: str = promise.get("name", "unnamed")

        try:
            if ptype == "output":
                actual = self._get_output_value(metric, prediction, prediction_proba)
                if actual is None:
                    return None
                if not self._evaluate(actual, operator, threshold):
                    return self._breach(name, ptype, actual, threshold, operator, promise)

            elif ptype == "latency":
                if latency_ms is None:
                    return None
                if not self._evaluate(latency_ms, operator, threshold):
                    return self._breach(name, ptype, latency_ms, threshold, operator, promise)

            elif ptype == "feature_range":
                feature_key: str = promise.get("feature_key") or metric
                val = features.get(feature_key)
                if val is None:
                    return None
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    return None
                if not self._evaluate(fval, operator, threshold):
                    return self._breach(name, ptype, fval, threshold, operator, promise)

            elif ptype == "distribution":
                return self._check_distribution(db, model_id, promise, prediction)

            elif ptype == "fairness":
                return self._check_fairness(db, model_id, promise, features, prediction)

        except Exception as e:
            logger.warning(
                f"promise_check_failed promise={name!r} type={ptype} "
                f"model_id={model_id} error={e}"
            )

        return None

    # ── Helper: build breach dict ──────────────────────────────────────────────

    @staticmethod
    def _breach(
        name: str,
        ptype: str,
        actual: Any,
        threshold: float,
        operator: str,
        promise: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "promise": name,
            "type": ptype,
            "actual": actual,
            "threshold": threshold,
            "operator": operator,
            "severity": promise.get("severity", "HIGH"),
            "action": promise.get("action", "alert"),
        }

    # ── Helper: extract output value ──────────────────────────────────────────

    @staticmethod
    def _get_output_value(
        metric: str,
        prediction: Any,
        prediction_proba: Optional[float],
    ) -> Optional[float]:
        if metric == "prediction_proba":
            return prediction_proba
        if metric == "prediction":
            try:
                return float(prediction)
            except (TypeError, ValueError):
                return None
        return None

    # ── Helper: operator evaluation ────────────────────────────────────────────

    @staticmethod
    def _evaluate(actual: float, operator: str, threshold: float) -> bool:
        """Returns True when the constraint IS satisfied (no breach)."""
        ops = {
            "lte": actual <= threshold,
            "gte": actual >= threshold,
            "lt":  actual <  threshold,
            "gt":  actual >  threshold,
            "eq":  actual == threshold,
            "neq": actual != threshold,
        }
        return ops.get(operator, True)

    # ── Distribution check ─────────────────────────────────────────────────────

    def _check_distribution(
        self,
        db: Session,
        model_id: str,
        promise: Dict[str, Any],
        prediction: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Check the rolling positive-prediction rate over a time window.
        e.g. "No more than 40% of predictions should be class=1 in 24h"
        Skips when fewer than 10 samples are available.
        """
        from app.db.models import PredictionLog

        window_hours: int = int(promise.get("window_hours", 24))
        threshold: float = float(promise.get("threshold", 0.5))
        operator: str = promise.get("operator", "lte")
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)

        try:
            rows = (
                db.query(PredictionLog.prediction)
                .filter(
                    PredictionLog.model_id == model_id,
                    PredictionLog.timestamp >= cutoff,
                )
                .all()
            )
            if len(rows) < 10:
                return None  # insufficient data

            _positive_vals = {"1", "true", "True", "1.0", "yes"}
            positive = sum(1 for (r,) in rows if str(r) in _positive_vals)
            rate = positive / len(rows)

            if not self._evaluate(rate, operator, threshold):
                return {
                    "promise": promise.get("name"),
                    "type": "distribution",
                    "actual": round(rate, 4),
                    "threshold": threshold,
                    "operator": operator,
                    "window_hours": window_hours,
                    "sample_count": len(rows),
                    "severity": promise.get("severity", "HIGH"),
                    "action": promise.get("action", "alert"),
                }
        except Exception as e:
            logger.warning(f"distribution_check_failed model_id={model_id} error={e}")

        return None

    # ── Fairness / demographic parity check ───────────────────────────────────

    def _check_fairness(
        self,
        db: Session,
        model_id: str,
        promise: Dict[str, Any],
        features: Dict[str, Any],
        prediction: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Check demographic parity difference over a rolling window.
        Compares positive prediction rates across protected attribute groups.
        Breach when max_rate - min_rate > threshold.
        Skips when fewer than 20 samples or fewer than 2 groups.
        """
        from app.db.models import PredictionLog

        protected: str = promise.get("protected_attribute", "")
        threshold: float = float(promise.get("threshold", 0.1))
        window_hours: int = int(promise.get("window_hours", 24))

        if not protected:
            return None

        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        try:
            rows = (
                db.query(PredictionLog.features, PredictionLog.prediction)
                .filter(
                    PredictionLog.model_id == model_id,
                    PredictionLog.timestamp >= cutoff,
                )
                .all()
            )
            if len(rows) < 20:
                return None

            _positive_vals = {"1", "true", "True", "1.0", "yes"}
            groups: Dict[str, List[bool]] = {}

            for feat_dict, pred in rows:
                feat_dict = feat_dict or {}
                group_key = feat_dict.get(protected)
                if group_key is None:
                    continue
                key = str(group_key)
                groups.setdefault(key, []).append(str(pred) in _positive_vals)

            if len(groups) < 2:
                return None

            rates = {
                k: sum(v) / len(v)
                for k, v in groups.items()
                if len(v) >= 5
            }
            if len(rates) < 2:
                return None

            max_rate = max(rates.values())
            min_rate = min(rates.values())
            dpd = round(max_rate - min_rate, 4)

            if dpd > threshold:
                return {
                    "promise": promise.get("name"),
                    "type": "fairness",
                    "actual": dpd,
                    "threshold": threshold,
                    "operator": "lte",
                    "protected_attribute": protected,
                    "group_rates": rates,
                    "severity": promise.get("severity", "CRITICAL"),
                    "action": promise.get("action", "alert"),
                }
        except Exception as e:
            logger.warning(f"fairness_check_failed model_id={model_id} error={e}")

        return None
