"""
Data Quality Validation Module.
Validates dataset quality before model training.
"""
import numpy as np
import hashlib
import logging

logger = logging.getLogger(__name__)


def validate_dataset(df, target_column=None, reference_df=None):
    """
    Comprehensive dataset quality validation.
    Returns a quality report with individual checks and overall score.
    """
    report = {
        "checks": {},
        "warnings": [],
        "quality_score": 0,
    }

    # 1. Missing values ratio
    missing = _check_missing_values(df)
    report["checks"]["missing_values"] = missing

    # 2. Duplicate rows
    duplicates = _check_duplicates(df)
    report["checks"]["duplicate_rows"] = duplicates

    # 3. Class imbalance (if target column specified)
    if target_column and target_column in df.columns:
        imbalance = _check_class_imbalance(df, target_column)
        report["checks"]["class_imbalance"] = imbalance
    else:
        report["checks"]["class_imbalance"] = {"ratio": None, "status": "skipped", "detail": "No target column specified"}

    # 4. Schema validation (if reference provided)
    if reference_df is not None:
        schema = _check_schema_match(df, reference_df)
        report["checks"]["schema_validation"] = schema
    else:
        report["checks"]["schema_validation"] = {"match": None, "status": "skipped"}

    # 5. Feature drift (if reference provided)
    if reference_df is not None:
        drift = _check_feature_drift(df, reference_df)
        report["checks"]["feature_drift"] = drift

    # 6. Constant columns
    constant = _check_constant_columns(df)
    report["checks"]["constant_columns"] = constant

    # 7. High cardinality
    cardinality = _check_high_cardinality(df)
    report["checks"]["high_cardinality"] = cardinality

    # 8. Outlier detection
    outliers = _check_outliers(df)
    report["checks"]["outliers"] = outliers

    # Compute quality score
    report["quality_score"] = _compute_quality_score(report["checks"])
    report["row_count"] = len(df)
    report["feature_count"] = len(df.columns)
    report["schema_hash"] = _compute_schema_hash(df)

    return report


def _check_missing_values(df):
    missing_ratios = (df.isnull().sum() / len(df)).to_dict()
    overall_missing = float(df.isnull().sum().sum() / (len(df) * len(df.columns)))
    problematic = {k: round(v, 4) for k, v in missing_ratios.items() if v > 0.01}

    status = "passed" if overall_missing < 0.05 else "warning" if overall_missing < 0.15 else "failed"
    return {
        "overall_missing_ratio": round(overall_missing, 4),
        "problematic_columns": problematic,
        "status": status,
    }


def _check_duplicates(df):
    dup_count = int(df.duplicated().sum())
    dup_ratio = round(dup_count / max(len(df), 1), 4)
    status = "passed" if dup_ratio < 0.01 else "warning" if dup_ratio < 0.1 else "failed"
    return {
        "duplicate_count": dup_count,
        "duplicate_ratio": dup_ratio,
        "status": status,
    }


def _check_class_imbalance(df, target_column):
    value_counts = df[target_column].value_counts()
    if len(value_counts) < 2:
        return {"ratio": 1.0, "status": "failed", "detail": "Only one class found"}

    majority = value_counts.iloc[0]
    minority = value_counts.iloc[-1]
    ratio = round(float(minority / majority), 4)

    status = "passed" if ratio > 0.4 else "warning" if ratio > 0.2 else "failed"
    return {
        "ratio": ratio,
        "class_distribution": {str(k): int(v) for k, v in value_counts.items()},
        "status": status,
    }


def _check_schema_match(df, reference_df):
    current_cols = set(df.columns)
    ref_cols = set(reference_df.columns)
    missing = list(ref_cols - current_cols)
    extra = list(current_cols - ref_cols)
    match = len(missing) == 0 and len(extra) == 0
    status = "passed" if match else "warning" if len(missing) == 0 else "failed"
    return {
        "match": match,
        "missing_columns": missing,
        "extra_columns": extra,
        "status": status,
    }


def _check_feature_drift(df, reference_df):
    """Simple PSI-based drift check for numeric columns."""
    common_cols = list(set(df.select_dtypes(include=[np.number]).columns) &
                       set(reference_df.select_dtypes(include=[np.number]).columns))
    drifted = {}
    for col in common_cols[:20]:  # Limit to 20 columns
        try:
            psi = _calculate_psi(reference_df[col].dropna().values, df[col].dropna().values)
            if psi > 0.1:
                drifted[col] = round(psi, 4)
        except Exception:
            pass
    status = "passed" if len(drifted) == 0 else "warning" if len(drifted) < 3 else "failed"
    return {"drifted_features": drifted, "drift_count": len(drifted), "status": status}


def _check_constant_columns(df):
    constant = [col for col in df.columns if df[col].nunique() <= 1]
    status = "passed" if len(constant) == 0 else "warning"
    return {"constant_columns": constant, "count": len(constant), "status": status}


def _check_high_cardinality(df, threshold=0.95):
    high_card = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        ratio = df[col].nunique() / max(len(df), 1)
        if ratio > threshold:
            high_card[col] = round(ratio, 4)
    status = "passed" if len(high_card) == 0 else "warning"
    return {"high_cardinality_columns": high_card, "count": len(high_card), "status": status}


def _check_outliers(df, z_threshold=3.0):
    outlier_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns[:20]:
        try:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                count = int((z_scores > z_threshold).sum())
                if count > 0:
                    outlier_counts[col] = count
        except Exception:
            pass
    total_outliers = sum(outlier_counts.values())
    status = "passed" if total_outliers < len(df) * 0.01 else "warning"
    return {"outlier_columns": outlier_counts, "total_outliers": total_outliers, "status": status}


def _compute_quality_score(checks):
    """Compute overall quality score (0-100) from individual checks."""
    score = 100
    for check_name, result in checks.items():
        status = result.get("status", "skipped")
        if status == "failed":
            score -= 20
        elif status == "warning":
            score -= 8
    return max(0, min(100, score))


def _compute_schema_hash(df):
    """Hash the column names + dtypes for schema tracking."""
    schema_str = "|".join(f"{col}:{dtype}" for col, dtype in zip(df.columns, df.dtypes))
    return hashlib.sha256(schema_str.encode()).hexdigest()[:32]


def _calculate_psi(expected, actual, bins=10):
    """Population Stability Index."""
    breakpoints = np.linspace(min(expected.min(), actual.min()),
                              max(expected.max(), actual.max()), bins + 1)
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 0.0001, None)
    actual_pct = np.clip(actual_pct, 0.0001, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
