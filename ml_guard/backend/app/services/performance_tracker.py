"""
performance_tracker.py — ML Guard Live Performance Degradation Tracker

Computes classification or regression metrics from labeled PredictionLogs,
compares against baseline, and generates degradation alerts.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import PerformanceSnapshot, PredictionLog
from app.services.ingestion_service import get_recent_predictions_df

logger = logging.getLogger(__name__)


# ─── Metric Computation ──────────────────────────────────────────────────────

def _compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, float]:
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                  recall_score, roc_auc_score, log_loss)
    metrics: Dict[str, float] = {}
    try:
        metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
        metrics["f1"] = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
        metrics["precision"] = round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
        metrics["recall"] = round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
        if y_proba is not None and len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
            metrics["log_loss"] = round(float(log_loss(y_true, y_proba)), 4)
    except Exception as e:
        logger.warning("classification_metric_error", error=str(e))
    return metrics


def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    metrics: Dict[str, float] = {}
    try:
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
        metrics["mae"] = round(float(mean_absolute_error(y_true, y_pred)), 4)
        metrics["r2"] = round(float(r2_score(y_true, y_pred)), 4)
        # MAPE — avoid division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            metrics["mape"] = round(float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100), 4)
    except Exception as e:
        logger.warning("regression_metric_error", error=str(e))
    return metrics


def _build_degradation_report(
    current: Dict[str, float],
    baseline: Dict[str, float],
    alert_threshold_pct: float = 5.0,
) -> Dict[str, Any]:
    """Compute delta and alert flag for each metric."""
    report: Dict[str, Any] = {}
    # Higher-is-better metrics
    higher_better = {"accuracy", "f1", "precision", "recall", "roc_auc", "r2"}

    for key, cur_val in current.items():
        base_val = baseline.get(key)
        if base_val is None:
            continue

        delta = cur_val - base_val
        pct_change = (delta / (abs(base_val) + 1e-9)) * 100

        alert = False
        if key in higher_better:
            alert = pct_change < -alert_threshold_pct
        else:
            # Lower is better: rmse, mae, log_loss, mape
            alert = pct_change > alert_threshold_pct

        report[key] = {
            "baseline": base_val,
            "current": cur_val,
            "delta": round(delta, 4),
            "pct_change": round(pct_change, 2),
            "alert": alert,
        }
    return report


# ─── Slice Analysis ──────────────────────────────────────────────────────────

def _analyze_slice(
    df: pd.DataFrame,
    slice_feature: str,
    task_type: str,
) -> Dict[str, Any]:
    """Compute metrics per slice value of a feature."""
    if slice_feature not in df.columns:
        return {"error": f"Feature '{slice_feature}' not found in prediction logs."}

    labeled = df.dropna(subset=["ground_truth"])
    if labeled.empty:
        return {"error": "No labeled predictions available."}

    slices: Dict[str, Dict] = {}
    for val, group in labeled.groupby(slice_feature):
        y_true = pd.to_numeric(group["ground_truth"], errors="coerce").dropna().values
        y_pred = pd.to_numeric(group["prediction"], errors="coerce").dropna().values

        if len(y_true) < 5:
            continue

        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]

        if task_type == "regression":
            m = _compute_regression_metrics(y_true, y_pred)
        else:
            y_pred_bin = (y_pred > 0.5).astype(int)
            m = _compute_classification_metrics(y_true, y_pred_bin, y_pred)

        slices[str(val)] = {**m, "sample_count": len(group)}

    return {"slice_feature": slice_feature, "slices": slices}


# ─── Main Tracker Class ───────────────────────────────────────────────────────

class PerformanceTracker:
    """
    Computes live performance metrics from PredictionLog ground-truth labels
    and generates degradation reports against stored baseline metrics.
    """

    def __init__(self, db: Session, model_id: str):
        self.db = db
        self.model_id = model_id

    def _get_task_type(self) -> str:
        """Retrieve stored task type for model (defaults to classification)."""
        snap = (
            self.db.query(PerformanceSnapshot)
            .filter(PerformanceSnapshot.model_id == self.model_id)
            .order_by(PerformanceSnapshot.computed_at.desc())
            .first()
        )
        return snap.task_type if snap else "classification"

    def compute_snapshot(
        self,
        window_hours: int = 24,
        task_type: Optional[str] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute a PerformanceSnapshot from labeled predictions in the window.
        Persists to DB and returns the result dict.
        """
        df = get_recent_predictions_df(self.db, self.model_id, hours=window_hours)
        if df.empty:
            return None

        labeled = df.dropna(subset=["ground_truth"])
        label_coverage_pct = round(len(labeled) / len(df) * 100, 2) if len(df) > 0 else 0.0

        task_type = task_type or self._get_task_type()

        if len(labeled) < 10:
            logger.warning("insufficient_labeled_data", model_id=self.model_id, labeled=len(labeled))
            return None

        y_true = pd.to_numeric(labeled["ground_truth"], errors="coerce").fillna(0).values
        y_pred_raw = pd.to_numeric(labeled["prediction"], errors="coerce").fillna(0).values
        y_proba = pd.to_numeric(labeled["prediction_proba"], errors="coerce").values

        if task_type == "regression":
            metrics = _compute_regression_metrics(y_true, y_pred_raw)
        else:
            y_pred = (y_pred_raw > 0.5).astype(int)
            metrics = _compute_classification_metrics(y_true, y_pred, y_proba if not np.all(np.isnan(y_proba)) else None)

        degradation_report = {}
        if baseline_metrics:
            degradation_report = _build_degradation_report(metrics, baseline_metrics)

        now = datetime.utcnow()
        window_start = now - timedelta(hours=window_hours)

        snap = PerformanceSnapshot(
            model_id=self.model_id,
            window_start=window_start,
            window_end=now,
            task_type=task_type,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            degradation_report=degradation_report,
            sample_count=len(df),
            labeled_count=len(labeled),
            label_coverage_pct=label_coverage_pct,
        )
        self.db.add(snap)
        self.db.commit()
        self.db.refresh(snap)

        return {
            "snapshot_id": str(snap.id),
            "model_id": self.model_id,
            "computed_at": snap.computed_at.isoformat(),
            "window_start": window_start.isoformat(),
            "window_end": now.isoformat(),
            "task_type": task_type,
            "metrics": metrics,
            "degradation_report": degradation_report,
            "sample_count": len(df),
            "labeled_count": len(labeled),
            "label_coverage_pct": label_coverage_pct,
        }

    def get_timeline(self, limit: int = 48) -> List[Dict[str, Any]]:
        """Return the last N performance snapshots as a timeline."""
        snaps = (
            self.db.query(PerformanceSnapshot)
            .filter(PerformanceSnapshot.model_id == self.model_id)
            .order_by(PerformanceSnapshot.computed_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "snapshot_id": str(s.id),
                "computed_at": s.computed_at.isoformat(),
                "metrics": s.metrics,
                "label_coverage_pct": s.label_coverage_pct,
                "sample_count": s.sample_count,
                "degradation_report": s.degradation_report,
            }
            for s in snaps
        ]

    def analyze_slice(self, slice_feature: str, window_hours: int = 24) -> Dict[str, Any]:
        """Compute performance broken down by a specific feature slice."""
        df = get_recent_predictions_df(self.db, self.model_id, hours=window_hours)
        if df.empty:
            return {"error": "No prediction data available."}
        task_type = self._get_task_type()
        return _analyze_slice(df, slice_feature, task_type)
