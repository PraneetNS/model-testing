"""
drift_analyzer.py — ML Guard Feature-Level Drift Monitor

Computes per-feature drift metrics for both numerical and categorical
features using KS test, PSI, Wasserstein distance, and chi-squared.
Writes results to DriftReport and triggers governance alerts on breach.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.models import DriftReport, PredictionLog
from app.db.session import SessionLocal
from app.services.ingestion_service import (
    get_recent_predictions_df,
    load_baseline_from_minio,
    store_baseline_to_minio,
)

logger = logging.getLogger(__name__)

# ─── Severity thresholds ────────────────────────────────────────────────────
KS_THRESHOLDS = {"LOW": 0.05, "MEDIUM": 0.10, "HIGH": 0.20, "CRITICAL": 0.30}
PSI_THRESHOLDS = {"LOW": 0.10, "MEDIUM": 0.20, "HIGH": 0.25, "CRITICAL": 0.40}


def _classify_severity(score: float, thresholds: dict) -> str:
    if score >= thresholds["CRITICAL"]:
        return "CRITICAL"
    if score >= thresholds["HIGH"]:
        return "HIGH"
    if score >= thresholds["MEDIUM"]:
        return "MEDIUM"
    if score >= thresholds["LOW"]:
        return "LOW"
    return "NONE"


def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    try:
        eps = 1e-6
        min_v = min(reference.min(), current.min())
        max_v = max(reference.max(), current.max())
        breaks = np.linspace(min_v, max_v, bins + 1)

        ref_counts, _ = np.histogram(reference, bins=breaks)
        cur_counts, _ = np.histogram(current, bins=breaks)

        ref_pct = (ref_counts / len(reference)) + eps
        cur_pct = (cur_counts / len(current)) + eps

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(round(psi, 6))
    except Exception:
        return 0.0


def _analyze_numerical_feature(
    name: str,
    ref: np.ndarray,
    cur: np.ndarray,
    method: str = "ks",
) -> Dict[str, Any]:
    """Full drift analysis for a numerical feature."""
    ks_stat, ks_p = stats.ks_2samp(ref, cur)
    psi = _compute_psi(ref, cur)
    wasserstein = float(stats.wasserstein_distance(ref, cur))

    ref_mean = float(np.mean(ref))
    cur_mean = float(np.mean(cur))
    mean_shift_pct = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-9) * 100

    if method == "psi":
        drift_score = psi
        thresholds = PSI_THRESHOLDS
        drift_detected = psi > PSI_THRESHOLDS["LOW"]
    else:  # default KS
        drift_score = float(ks_stat)
        thresholds = KS_THRESHOLDS
        drift_detected = ks_p < 0.05

    severity = _classify_severity(drift_score, thresholds)

    return {
        "feature_name": name,
        "type": "numerical",
        "drift_detected": drift_detected,
        "drift_score": round(drift_score, 6),
        "method": method,
        "p_value": round(float(ks_p), 6),
        "ks_statistic": round(float(ks_stat), 6),
        "psi": round(psi, 6),
        "wasserstein": round(wasserstein, 6),
        "reference_mean": round(ref_mean, 4),
        "current_mean": round(cur_mean, 4),
        "reference_std": round(float(np.std(ref)), 4),
        "current_std": round(float(np.std(cur)), 4),
        "mean_shift_pct": round(mean_shift_pct, 2),
        "severity": severity,
        "n_reference": len(ref),
        "n_current": len(cur),
    }


def _analyze_categorical_feature(
    name: str,
    ref_series: pd.Series,
    cur_series: pd.Series,
) -> Dict[str, Any]:
    """Full drift analysis for a categorical feature."""
    ref_cats = set(ref_series.unique())
    cur_cats = set(cur_series.unique())
    new_categories = list(cur_cats - ref_cats)

    # Align categories for chi2
    all_cats = sorted(ref_cats | cur_cats)
    ref_counts = [ref_series.value_counts().get(c, 0) for c in all_cats]
    cur_counts = [cur_series.value_counts().get(c, 0) for c in all_cats]

    drift_score = 0.0
    p_value = 1.0
    drift_detected = False
    if sum(ref_counts) > 0 and sum(cur_counts) > 0:
        try:
            chi2, p_value, _, _ = stats.chi2_contingency(
                [ref_counts, cur_counts]
            )
            drift_score = float(chi2)
            drift_detected = p_value < 0.05
        except Exception:
            pass

    ref_dist = {c: round(ref_series.value_counts(normalize=True).get(c, 0.0), 4)
                for c in all_cats}
    cur_dist = {c: round(cur_series.value_counts(normalize=True).get(c, 0.0), 4)
                for c in all_cats}

    severity = "HIGH" if drift_detected and new_categories else ("MEDIUM" if drift_detected else "NONE")

    return {
        "feature_name": name,
        "type": "categorical",
        "drift_detected": drift_detected,
        "drift_score": round(drift_score, 4),
        "method": "chi2",
        "p_value": round(float(p_value), 6),
        "new_categories": new_categories,
        "reference_distribution": ref_dist,
        "current_distribution": cur_dist,
        "severity": severity,
        "n_reference": len(ref_series),
        "n_current": len(cur_series),
    }


class DriftAnalyzer:
    """
    Orchestrates per-feature drift analysis between a reference distribution
    and a current production window of PredictionLog data.
    """

    SKIP_COLS = {"log_id", "timestamp", "prediction", "prediction_proba",
                 "ground_truth", "latency_ms", "environment"}

    def __init__(self, db: AsyncSession, model_id: str, method: str = "ks"):
        self.db = db
        self.model_id = model_id
        self.method = method

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c not in self.SKIP_COLS]

    async def analyze(
        self,
        window_hours: int = 24,
        min_samples: int = 30,
    ) -> Optional[Dict[str, Any]]:
        """
        Run full drift analysis for the model. Returns a structured report dict
        or None if insufficient data.
        """
        # 1. Load current window
        current_df = get_recent_predictions_df(self.db, self.model_id, hours=window_hours)
        if current_df.empty or len(current_df) < min_samples:
            logger.warning("drift_insufficient_data", model_id=self.model_id, count=len(current_df))
            return None

        # 2. Load reference baseline
        ref_df = load_baseline_from_minio(self.model_id)
        if ref_df is None or ref_df.empty:
            # Auto-bootstrap: first run stores current window as reference
            feature_cols = self._get_feature_cols(current_df)
            store_baseline_to_minio(self.model_id, current_df[feature_cols])
            logger.info("drift_baseline_bootstrapped", model_id=self.model_id)
            return None

        feature_cols = [c for c in self._get_feature_cols(current_df) if c in ref_df.columns]
        if not feature_cols:
            return None

        feature_results: List[Dict[str, Any]] = []
        drift_scores: List[float] = []

        for col in feature_cols:
            ref_series = ref_df[col].dropna()
            cur_series = current_df[col].dropna()

            if len(ref_series) < 5 or len(cur_series) < 5:
                continue

            # Infer type
            if pd.api.types.is_numeric_dtype(ref_series):
                result = _analyze_numerical_feature(
                    col, ref_series.values, cur_series.values, method=self.method
                )
            else:
                result = _analyze_categorical_feature(col, ref_series, cur_series)

            feature_results.append(result)
            drift_scores.append(result["drift_score"])

        if not feature_results:
            return None

        overall_drift_score = float(np.mean(drift_scores))
        drift_detected = any(r["drift_detected"] for r in feature_results)
        max_severity = max(
            (r["severity"] for r in feature_results if r["drift_detected"]),
            default="NONE",
            key=lambda s: {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}[s]
        )

        now = datetime.utcnow()
        window_start = now - timedelta(hours=window_hours)

        # 3. Persist to DB
        report = DriftReport(
            model_id=self.model_id,
            reference_window_start=None,  # baseline is static
            reference_window_end=None,
            current_window_start=window_start,
            current_window_end=now,
            feature_results=feature_results,
            overall_drift_score=overall_drift_score,
            drift_detected=drift_detected,
            method=self.method,
            sample_count=len(current_df),
        )
        self.db.add(report)
        await self.db.commit()
        await self.db.refresh(report)

        # 4. Trigger governance audit if threshold breached
        if drift_detected and max_severity in ("HIGH", "CRITICAL"):
            self._trigger_governance_audit(str(report.id), max_severity)
            report.alert_triggered = True
            await self.db.commit()

        return {
            "report_id": str(report.id),
            "model_id": self.model_id,
            "created_at": report.created_at.isoformat(),
            "drift_detected": drift_detected,
            "overall_drift_score": round(overall_drift_score, 6),
            "max_severity": max_severity,
            "method": self.method,
            "feature_count": len(feature_results),
            "feature_results": feature_results,
            "sample_count": len(current_df),
        }

    def _trigger_governance_audit(self, report_id: str, severity: str) -> None:
        """Dispatch auto-governance audit when drift crosses threshold."""
        try:
            from app.core.celery_app import celery_app
            celery_app.send_task(
                "app.tasks.run_governance_audit",
                kwargs={
                    "model_id": self.model_id,
                    "trigger": "auto_drift_alert",
                    "trigger_report_id": report_id,
                    "severity": severity,
                },
            )
            logger.info("governance_audit_triggered", model_id=self.model_id, severity=severity)
        except Exception as e:
            logger.warning("governance_trigger_failed", error=str(e))

    def get_history(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Return the last N drift reports for this model."""
        reports = (
            self.db.query(DriftReport)
            .filter(DriftReport.model_id == self.model_id)
            .order_by(DriftReport.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "report_id": str(r.id),
                "created_at": r.created_at.isoformat(),
                "drift_detected": r.drift_detected,
                "overall_drift_score": r.overall_drift_score,
                "method": r.method,
                "sample_count": r.sample_count,
                "alert_triggered": r.alert_triggered,
            }
            for r in reports
        ]

    def get_feature_timeline(self, feature_name: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Return per-feature drift score timeline for sparkline rendering."""
        reports = (
            self.db.query(DriftReport)
            .filter(DriftReport.model_id == self.model_id)
            .order_by(DriftReport.created_at.desc())
            .limit(limit)
            .all()
        )
        results = []
        for r in reports:
            for fr in (r.feature_results or []):
                if fr.get("feature_name") == feature_name:
                    results.append({
                        "timestamp": r.created_at.isoformat(),
                        "drift_score": fr.get("drift_score", 0),
                        "drift_detected": fr.get("drift_detected", False),
                        "severity": fr.get("severity", "NONE"),
                    })
                    break
        return results
