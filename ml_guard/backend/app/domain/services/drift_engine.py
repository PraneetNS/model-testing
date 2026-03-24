import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from typing import Dict, Any, List, Optional
import structlog

logger = structlog.get_logger(__name__)

class DriftEngine:
    """
    Statistically grounded Drift Detection Engine.
    Implements PSI, KS Test, and Chi-Square for enterprise-grade monitoring.
    """

    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculates Population Stability Index (PSI).
        Formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
        """
        def scale_range(data, min_val, max_val):
            return (data - min_val) / (max_val - min_val)

        # Handle constant data
        if np.all(expected == expected[0]) and np.all(actual == actual[0]):
            return 0.0 if expected[0] == actual[0] else 1.0

        breakpoints = np.linspace(0, 1, buckets + 1)
        
        # Min/Max for joint scaling
        all_min = min(expected.min(), actual.min())
        all_max = max(expected.max(), actual.max())
        
        e_scaled = scale_range(expected, all_min, all_max)
        a_scaled = scale_range(actual, all_min, all_max)

        expected_percents = np.histogram(e_scaled, bins=breakpoints, density=True)[0] / buckets
        actual_percents = np.histogram(a_scaled, bins=breakpoints, density=True)[0] / buckets

        # Replace 0s to avoid div by zero/log errors
        expected_percents = np.clip(expected_percents, 0.0001, 1.0)
        actual_percents = np.clip(actual_percents, 0.0001, 1.0)

        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return float(psi_value)

    @staticmethod
    def calculate_ks_test(expected: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
        """
        Kolmogorov-Smirnov Test for numeric feature drift.
        """
        statistic, p_value = ks_2samp(expected, actual)
        return {"statistic": float(statistic), "p_value": float(p_value)}

    @staticmethod
    def calculate_categorical_drift(expected: pd.Series, actual: pd.Series) -> Dict[str, float]:
        """
        Chi-Square Independence Test for categorical features.
        """
        # Combine counts
        e_counts = expected.value_counts()
        a_counts = actual.value_counts()
        
        # Align indexes
        all_cats = e_counts.index.union(a_counts.index)
        e_aligned = e_counts.reindex(all_cats, fill_value=0) + 1 # Laplace smoothing
        a_aligned = a_counts.reindex(all_cats, fill_value=0) + 1
        
        contingency = np.array([e_aligned.values, a_aligned.values])
        chi2, p, _, _ = chi2_contingency(contingency)
        
        return {"chi2": float(chi2), "p_value": float(p)}

    @staticmethod
    def calculate_correlation_shift(ref_df: pd.DataFrame, prod_df: pd.DataFrame) -> float:
        """Phase 3: Correlation matrix shift score using Frobenius norm"""
        num_cols = ref_df.select_dtypes(include=[np.number]).columns
        valid_cols = [c for c in num_cols if c in prod_df.columns]
        if len(valid_cols) < 2:
            return 0.0
        ref_corr = ref_df[valid_cols].corr().fillna(0).values
        prod_corr = prod_df[valid_cols].corr().fillna(0).values
        return float(np.linalg.norm(ref_corr - prod_corr, ord='fro'))

    def detect_drift(self, reference_df: pd.DataFrame, production_df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Full drift analysis between two datasets.
        """
        drift_report = {}
        overall_drift_score = 0
        total_features = 0

        # Phase 3: Correlation Matrix shift score
        correlation_shift = self.calculate_correlation_shift(reference_df, production_df)

        # Target distribution drift
        target_drift = None
        if target_column and target_column in reference_df.columns and target_column in production_df.columns:
            if np.issubdtype(reference_df[target_column].dtype, np.number):
                target_drift = self.calculate_ks_test(reference_df[target_column].values, production_df[target_column].values)
            else:
                target_drift = self.calculate_categorical_drift(reference_df[target_column], production_df[target_column])

        for col in reference_df.columns:
            if col not in production_df.columns or col == target_column:
                continue
            
            total_features += 1
            feature_drift = {"status": "stable"}
            
            # Numeric Features
            if np.issubdtype(reference_df[col].dtype, np.number):
                psi = self.calculate_psi(reference_df[col].values, production_df[col].values)
                ks = self.calculate_ks_test(reference_df[col].values, production_df[col].values)
                
                feature_drift.update({
                    "psi": round(psi, 4),
                    "ks_p_value": round(ks["p_value"], 4),
                    "type": "numeric"
                })
                
                if psi > 0.2 or ks["p_value"] < 0.05:
                    feature_drift["status"] = "drifted"
                    overall_drift_score += 1
            
            # Categorical Features
            else:
                chi2 = self.calculate_categorical_drift(reference_df[col], production_df[col])
                feature_drift.update({
                    "chi2_p_value": round(chi2["p_value"], 4),
                    "type": "categorical"
                })
                
                if chi2["p_value"] < 0.05:
                    feature_drift["status"] = "drifted"
                    overall_drift_score += 1
            
            drift_report[col] = feature_drift

        risk_score = (overall_drift_score / total_features) * 100 if total_features > 0 else 0
        if correlation_shift > 5.0:  # arbitrary threshold for shift score
            risk_score += 10
            
        severity = "High" if risk_score > 30 else "Medium" if risk_score > 10 else "Low"
        alert_triggers = [col for col, data in drift_report.items() if data['status'] == 'drifted']

        return {
            "risk_score": round(min(risk_score, 100.0), 2),
            "severity": severity,
            "feature_drift": drift_report,
            "correlation_shift_score": round(correlation_shift, 4),
            "target_drift": target_drift,
            "summary": {
                "total_features": total_features,
                "drifted_features": overall_drift_score,
                "alert_triggers": alert_triggers
            }
        }

    @staticmethod
    def top_drifted_features(drift_report: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Extracts, ranks, and tags the top-N most drifted features by PSI.

        Severity thresholds:
          PSI > 0.25 → CRITICAL
          PSI > 0.15 → WARNING
          else       → STABLE

        Returns a list sorted descending by PSI, length ≤ top_n.
        """
        ranked = []
        for feature, data in drift_report.items():
            if not isinstance(data, dict):
                continue
            psi = data.get("psi")
            if psi is None:
                continue
            psi = float(psi)
            if psi > 0.25:
                sev = "CRITICAL"
            elif psi > 0.15:
                sev = "WARNING"
            else:
                sev = "STABLE"
            ranked.append({
                "feature":  feature,
                "psi":      round(psi, 4),
                "severity": sev,
            })

        ranked.sort(key=lambda x: x["psi"], reverse=True)
        return ranked[:top_n]

