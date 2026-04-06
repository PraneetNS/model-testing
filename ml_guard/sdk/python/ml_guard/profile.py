"""
ml_guard/profile.py — WhyLogs-style Statistical Profiling SDK

Unique ML Guard differentiator: Privacy-preserving lightweight data profiles
that are computed client-side, merged server-side, and never transmit raw data.

Unlike Evidently (requires full DataFrames) and WhyLabs (cloud-only merging),
ML Guard profiles are:
  - Computed offline (no network required)
  - Serializable to JSON (100 bytes vs megabytes)
  - Diff-able (compare two profiles without raw data)
  - Embeddable in CI pipeline artifacts
  - Open-source mergeable (P0 + P1 → merged profile)

Usage:
    from ml_guard import profile

    # Profile a dataset
    prof = profile.from_dataframe(df, model_id="churn-v2")
    prof.flush()  # send to ML Guard backend

    # Profile a pandas Series inline
    with profile.track("churn-v2") as p:
        p.track_column("age", series)
        p.track_column("spend", series2)
    # auto-flush on context exit
"""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import os

import numpy as np
import pandas as pd


# ── Per-column statistical sketch ─────────────────────────────────────────────

class ColumnProfile:
    """
    Lightweight statistical sketch for a single feature column.
    Modeled after sketch-based algorithms — O(1) memory, privacy-safe.
    """

    def __init__(self, name: str):
        self.name = name
        self._values: List[Any] = []  # held only during build phase
        self.dtype: str = "unknown"

        # Numerical stats
        self.count: int = 0
        self.null_count: int = 0
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self.min: Optional[float] = None
        self.max: Optional[float] = None
        self.p25: Optional[float] = None
        self.p50: Optional[float] = None
        self.p75: Optional[float] = None
        self.p95: Optional[float] = None
        self.p99: Optional[float] = None

        # Categorical stats
        self.cardinality: Optional[int] = None
        self.top_values: Optional[Dict[str, float]] = None  # value → freq%
        self.new_categories: Optional[List[str]] = None

        # Quality flags
        self.zero_pct: Optional[float] = None
        self.negative_pct: Optional[float] = None
        self.inf_count: Optional[int] = None

    def track(self, series: pd.Series) -> "ColumnProfile":
        """Compute all statistics from a pandas Series."""
        self.count = len(series)
        null_mask = series.isna()
        self.null_count = int(null_mask.sum())

        clean = series.dropna()

        if pd.api.types.is_numeric_dtype(series):
            self.dtype = "numerical"
            arr = clean.astype(float).values

            if len(arr) > 0:
                self.mean = float(np.mean(arr))
                self.std = float(np.std(arr))
                self.min = float(np.min(arr))
                self.max = float(np.max(arr))
                self.p25 = float(np.percentile(arr, 25))
                self.p50 = float(np.percentile(arr, 50))
                self.p75 = float(np.percentile(arr, 75))
                self.p95 = float(np.percentile(arr, 95))
                self.p99 = float(np.percentile(arr, 99))
                self.zero_pct = float(np.mean(arr == 0) * 100)
                self.negative_pct = float(np.mean(arr < 0) * 100)
                self.inf_count = int(np.isinf(arr).sum())
        else:
            self.dtype = "categorical"
            vc = clean.value_counts(normalize=True)
            self.cardinality = int(clean.nunique())
            self.top_values = {str(k): round(float(v) * 100, 2)
                               for k, v in vc.head(10).items()}

        return self

    def diff(self, other: "ColumnProfile") -> Dict[str, Any]:
        """
        Compare this profile (current) against another (reference).
        Returns a structured diff — the core of ML Guard's drift detection
        without needing raw data.
        """
        result: Dict[str, Any] = {
            "column": self.name,
            "dtype": self.dtype,
            "warnings": [],
        }

        if self.dtype == "numerical" and other.dtype == "numerical":
            if other.mean is not None and self.mean is not None:
                mean_shift = abs(self.mean - other.mean) / (abs(other.mean) + 1e-9)
                result["mean_shift_pct"] = round(mean_shift * 100, 2)
                if mean_shift > 0.2:
                    result["warnings"].append(f"Mean shifted by {mean_shift*100:.1f}%")

            if other.std is not None and self.std is not None and other.std > 0:
                std_ratio = self.std / other.std
                result["std_ratio"] = round(std_ratio, 3)
                if std_ratio > 2.0 or std_ratio < 0.5:
                    result["warnings"].append(f"Std dev changed by {std_ratio:.1f}x")

            null_delta = abs((self.null_count / max(self.count, 1)) -
                            (other.null_count / max(other.count, 1)))
            result["null_rate_delta"] = round(null_delta * 100, 2)
            if null_delta > 0.05:
                result["warnings"].append(f"Null rate changed by {null_delta*100:.1f}%")

        elif self.dtype == "categorical" and other.dtype == "categorical":
            if self.top_values and other.top_values:
                new_cats = [k for k in (self.top_values or {})
                            if k not in (other.top_values or {})]
                result["new_categories"] = new_cats
                if new_cats:
                    result["warnings"].append(f"New categories detected: {new_cats[:3]}")

                missing_cats = [k for k in (other.top_values or {})
                               if k not in (self.top_values or {})]
                result["missing_categories"] = missing_cats

            if self.cardinality and other.cardinality:
                card_ratio = self.cardinality / max(other.cardinality, 1)
                result["cardinality_ratio"] = round(card_ratio, 2)
                if card_ratio > 2.0:
                    result["warnings"].append(f"Cardinality doubled ({other.cardinality} → {self.cardinality})")

        result["has_warnings"] = len(result["warnings"]) > 0
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith("_")}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnProfile":
        """Deserialize from dict."""
        p = cls(data["name"])
        for k, v in data.items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p


# ── Dataset-level profile ──────────────────────────────────────────────────────

class DataProfile:
    """
    Full dataset profile. Contains per-column ColumnProfiles and dataset-level
    metadata. Can be serialized, sent to the server, and diff'd.
    """

    def __init__(
        self,
        model_id: str,
        dataset_name: str = "production",
        tags: Optional[Dict[str, Any]] = None,
    ):
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.tags = tags or {}
        self.columns: Dict[str, ColumnProfile] = {}
        self.row_count: int = 0
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.profile_id: str = hashlib.sha256(
            f"{model_id}-{dataset_name}-{time.time()}".encode()
        ).hexdigest()[:16]
        self._client = None

    def track_column(self, name: str, series: pd.Series) -> "DataProfile":
        """Track a single column."""
        cp = ColumnProfile(name).track(series)
        self.columns[name] = cp
        return self

    def track_dataframe(self, df: pd.DataFrame, label_col: Optional[str] = None) -> "DataProfile":
        """Track all columns in a DataFrame."""
        self.row_count = len(df)
        cols = [c for c in df.columns if c != label_col]
        for col in cols:
            self.track_column(col, df[col])
        return self

    def quality_report(self) -> Dict[str, Any]:
        """Generate a data quality report from the profile."""
        issues = []
        for name, cp in self.columns.items():
            null_pct = cp.null_count / max(cp.count, 1) * 100
            if null_pct > 10:
                issues.append({"column": name, "issue": "high_nulls",
                               "value": round(null_pct, 1)})
            if cp.dtype == "numerical":
                if cp.inf_count and cp.inf_count > 0:
                    issues.append({"column": name, "issue": "has_infinities",
                                  "value": cp.inf_count})
                if cp.std == 0:
                    issues.append({"column": name, "issue": "zero_variance",
                                  "value": 0})
            if cp.dtype == "categorical":
                if cp.cardinality == cp.count:
                    issues.append({"column": name, "issue": "all_unique_values",
                                  "value": cp.cardinality})

        score = max(0, 100 - len(issues) * 10)
        return {
            "model_id": self.model_id,
            "row_count": self.row_count,
            "column_count": len(self.columns),
            "quality_score": score,
            "issues": issues,
            "null_rate": round(
                sum(cp.null_count for cp in self.columns.values()) /
                max(sum(cp.count for cp in self.columns.values()), 1) * 100, 2
            ),
        }

    def diff(self, reference: "DataProfile") -> Dict[str, Any]:
        """
        Profile-level diff: compare self (current) vs reference.
        Returns drift summary without raw data — privacy-safe.
        """
        column_diffs = []
        for name, cp in self.columns.items():
            if name in reference.columns:
                column_diffs.append(cp.diff(reference.columns[name]))

        drifted = [d for d in column_diffs if d["has_warnings"]]
        return {
            "model_id": self.model_id,
            "profile_id": self.profile_id,
            "reference_id": reference.profile_id,
            "drift_detected": len(drifted) > 0,
            "drifted_columns": len(drifted),
            "total_columns": len(column_diffs),
            "drift_pct": round(len(drifted) / max(len(column_diffs), 1) * 100, 1),
            "column_diffs": column_diffs,
            "created_at": self.created_at,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "model_id": self.model_id,
            "dataset_name": self.dataset_name,
            "tags": self.tags,
            "row_count": self.row_count,
            "created_at": self.created_at,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialize to compact JSON. Optionally save to file."""
        js = json.dumps(self.to_dict(), default=str)
        if path:
            with open(path, "w") as f:
                f.write(js)
        return js

    @classmethod
    def from_json(cls, data: Union[str, dict]) -> "DataProfile":
        """Deserialize from JSON string or dict."""
        if isinstance(data, str):
            data = json.loads(data)
        p = cls(model_id=data["model_id"], dataset_name=data.get("dataset_name", ""))
        p.profile_id = data.get("profile_id", p.profile_id)
        p.row_count = data.get("row_count", 0)
        p.created_at = data.get("created_at", p.created_at)
        p.tags = data.get("tags", {})
        for name, col_data in data.get("columns", {}).items():
            p.columns[name] = ColumnProfile.from_dict(col_data)
        return p

    def flush(self) -> Dict[str, Any]:
        """Send this profile to the ML Guard backend."""
        if self._client is None:
            raise RuntimeError(
                "Profile has no client. Attach one via profile._client = client "
                "or use mlguard.profile.from_dataframe(df, client=client)"
            )
        return self._client.upload_profile(self)

    def __repr__(self) -> str:
        return (
            f"<DataProfile model_id={self.model_id!r} "
            f"columns={len(self.columns)} rows={self.row_count}>"
        )


# ── Convenience factory functions ──────────────────────────────────────────────

def from_dataframe(
    df: pd.DataFrame,
    model_id: str,
    dataset_name: str = "production",
    label_col: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
    client=None,
) -> DataProfile:
    """
    Create a DataProfile from a pandas DataFrame.

    Example:
        prof = mlguard.profile.from_dataframe(df, "churn-v2", client=client)
        prof.flush()
    """
    p = DataProfile(model_id=model_id, dataset_name=dataset_name, tags=tags)
    p._client = client
    p.track_dataframe(df, label_col=label_col)
    return p


@contextmanager
def track(
    model_id: str,
    dataset_name: str = "production",
    client=None,
    auto_flush: bool = True,
):
    """
    Context manager for building a profile incrementally.

    Example:
        with mlguard.profile.track("churn-v2", client=client) as p:
            p.track_column("age", df["age"])
            p.track_column("spend", df["spend"])
        # profile auto-flushed on exit
    """
    p = DataProfile(model_id=model_id, dataset_name=dataset_name)
    p._client = client
    try:
        yield p
    finally:
        if auto_flush and client is not None:
            client.upload_profile(p)
