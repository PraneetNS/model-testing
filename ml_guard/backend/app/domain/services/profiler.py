from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import structlog
from app.domain.services.fingerprinting import FingerprintingService
from app.domain.services.detector import ModelDetector # I'll need to move/sync this or create a backend version

logger = structlog.get_logger(__name__)

class ModelProfiler:
    """
    Intelligent Model Profiler for ML Guard.
    Analyzes model + dataset to recommend governance policies.
    """
    
    PROTECTED_ATTRIBUTES = ["gender", "age", "race", "ethnicity", "religion", "disability", "nationality"]

    def profile_artifacts(
        self, 
        model: Any, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Dict[str, Any]:
        """
        Calculates data characteristics and recommends a governance profile.
        """
        # 1. Detect Model Characteristics
        # Simple detection (can be expanded with ModelDetector logic)
        model_name = str(type(model)).lower()
        model_type = "classification" if "classifier" in model_name else "regression"
        
        # 2. Analyze Dataset
        row_count = len(df)
        feature_count = len(df.columns) - 1
        missing_pct = (df.isnull().sum().sum() / (df.size)) * 100
        
        feature_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Class Imbalance (only for classification)
        imbalance_ratio = 1.0
        if model_type == "classification" and target_column in df.columns:
            counts = df[target_column].value_counts()
            if len(counts) > 1:
                imbalance_ratio = counts.max() / counts.min()

        # Detect Protected Attributes
        detected_protected = [col for col in df.columns if col.lower() in self.PROTECTED_ATTRIBUTES]

        profile = {
            "model_type": model_type,
            "dataset_stats": {
                "rows": row_count,
                "features": feature_count,
                "missing_pct": round(missing_pct, 2),
                "imbalance_ratio": round(imbalance_ratio, 2),
                "protected_attributes": detected_protected
            }
        }

        # 3. Suggest Risk Level
        risk_level, risk_reason = self._estimate_risk(profile)
        profile["suggested_risk"] = risk_level
        profile["risk_reason"] = risk_reason

        # 4. Generate Recommended Tests
        profile["recommended_tests"] = self._generate_test_suite(profile, target_column)
        
        # 5. Recommended Scoring weights
        profile["scoring_profile"] = self._generate_scoring_weights(risk_level)

        return profile

    def _estimate_risk(self, profile: Dict[str, Any]) -> Tuple[str, str]:
        stats = profile["dataset_stats"]
        
        if stats["protected_attributes"]:
            return "Critical", "Protected attributes detected; high bias risk profile."
        
        if stats["imbalance_ratio"] > 5.0:
            return "High", "Significant class imbalance may lead to biased predictions."
        
        if stats["missing_pct"] > 10.0:
            return "Medium", "High missing value percentage may reduce model reliability."
            
        if stats["rows"] < 1000:
            return "Medium", "Small dataset size; potential for overfitting."
            
        return "Low", "Standard profile with no significant risk indicators."

    def _generate_test_suite(self, profile: Dict[str, Any], target: str) -> List[Dict[str, Any]]:
        tests = []
        stats = profile["dataset_stats"]
        model_type = profile["model_type"]

        # Performance Basics
        tests.append({"type": "accuracy_threshold", "severity": "critical", "config": {"threshold": 0.8}})
        
        # Bias tests (Dynamic)
        for attr in stats["protected_attributes"]:
            tests.append({
                "type": "disparate_impact", 
                "severity": "critical", 
                "config": {"protected_attribute": attr, "threshold": 0.8}
            })

        # Data Quality
        if stats["missing_pct"] > 0:
            tests.append({"type": "missing_values", "severity": "high", "config": {"threshold": 0.05}})
            
        # Drift checks (Standard)
        tests.append({"type": "psi_drift", "severity": "critical", "config": {"threshold": 0.2}})
        
        # Robustness
        if profile["suggested_risk"] in ["High", "Critical"]:
            tests.append({"type": "input_perturbation", "severity": "high", "config": {"noise": 0.01}})

        return tests

    def _generate_scoring_weights(self, risk_level: str) -> Dict[str, float]:
        """Generates weights for the RiskEngine."""
        if risk_level == "Critical":
            return {"critical": 20.0, "high": 10.0, "medium": 5.0, "low": 1.0}
        elif risk_level == "High":
            return {"critical": 15.0, "high": 7.5, "medium": 3.0, "low": 1.0}
        else:
            return {"critical": 10.0, "high": 5.0, "medium": 2.5, "low": 1.0}

    def create_baseline(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Phase 2: Create dataset baseline (histograms, percentiles)"""
        baselines = []
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                counts, edges = np.histogram(df[col].dropna(), bins=10)
                baselines.append({
                    "feature_name": col,
                    "distribution_type": "numeric",
                    "histogram_bins": {"counts": counts.tolist(), "edges": edges.tolist()},
                    "percentiles": {
                        "p10": np.nanpercentile(df[col], 10),
                        "p25": np.nanpercentile(df[col], 25),
                        "p50": np.nanpercentile(df[col], 50),
                        "p75": np.nanpercentile(df[col], 75),
                        "p90": np.nanpercentile(df[col], 90)
                    }
                })
            else:
                counts = df[col].value_counts(normalize=True).to_dict()
                baselines.append({
                    "feature_name": col,
                    "distribution_type": "categorical",
                    "histogram_bins": counts,
                    "percentiles": {}
                })
        return baselines
