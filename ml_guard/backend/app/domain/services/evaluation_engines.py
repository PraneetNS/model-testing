from typing import Dict, Any, List
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

class BiasEngine:
    """
    ML Fairness & Bias Auditor.
    Implements Statistical Parity, Equal Opportunity, and Disparate Impact.
    """

    @staticmethod
    def audit_fairness(
        model: Any, 
        df: pd.DataFrame, 
        target_col: str, 
        protected_attr: str, 
        privileged_group: Any,
        unprivileged_group: Any
    ) -> Dict[str, Any]:
        """
        Calculates fairness metrics.
        - Statistical Parity Difference (SPD)
        - Disparate Impact Ratio (DI)
        - Equal Opportunity Difference (EOD)
        """
        try:
            X = df.drop(columns=[target_col])
            y_true = df[target_col]
            y_pred = model.predict(X)
            
            # Mask for protected groups
            mask_priv = df[protected_attr] == privileged_group
            mask_unpriv = df[protected_attr] == unprivileged_group
            
            # 1. Statistical Parity Difference
            # P(Y^=1 | unprivileged) - P(Y^=1 | privileged)
            prob_unpriv = np.mean(y_pred[mask_unpriv] == 1)
            prob_priv = np.mean(y_pred[mask_priv] == 1)
            spd = prob_unpriv - prob_priv
            
            # 2. Disparate Impact Ratio
            # P(Y^=1 | unprivileged) / P(Y^=1 | privileged)
            di = prob_unpriv / prob_priv if prob_priv > 0 else 1.0
            
            # 3. Equal Opportunity Difference (for positive classes)
            # TPR_unpriv - TPR_priv
            tpr_unpriv = np.mean(y_pred[mask_unpriv & (y_true == 1)] == 1)
            tpr_priv = np.mean(y_pred[mask_priv & (y_true == 1)] == 1)
            eod = tpr_unpriv - tpr_priv

            return {
                "statistical_parity_difference": round(float(spd), 4),
                "disparate_impact_ratio": round(float(di), 4),
                "equal_opportunity_difference": round(float(eod), 4),
                "status": "pass" if abs(spd) < 0.1 and 0.8 < di < 1.25 else "fail",
                "risk": "High" if abs(spd) > 0.2 or di < 0.7 else "Low"
            }
        except Exception as e:
            logger.error("Bias audit failed", error=str(e), attr=protected_attr)
            return {"error": str(e), "status": "error"}

class RobustnessEngine:
    """
    Model Sensitivity & Adversarial Robustness Tester.
    """

    @staticmethod
    def test_perturbation(model: Any, X: pd.DataFrame, noise_level: float = 0.05) -> Dict[str, Any]:
        """
        Measures stability by injecting Gaussian noise.
        Flip Rate = % of predictions that change after noise injection.
        """
        try:
            # Baseline predictions
            orig_preds = model.predict(X)
            
            # Inject noise into numeric columns
            X_noisy = X.copy()
            numeric_cols = X_noisy.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                std = X_noisy[col].std()
                noise = np.random.normal(0, std * noise_level, size=len(X_noisy))
                X_noisy[col] = X_noisy[col] + noise
                
            noisy_preds = model.predict(X_noisy)
            
            # Calculate Flip Rate
            flips = np.sum(orig_preds != noisy_preds)
            flip_rate = flips / len(orig_preds)
            
            return {
                "noise_level": noise_level,
                "flip_rate": round(float(flip_rate), 4),
                "stability_index": round(1.0 - float(flip_rate), 4),
                "status": "pass" if flip_rate < 0.1 else "warn" if flip_rate < 0.25 else "fail"
            }
        except Exception as e:
            logger.error("Robustness test failed", error=str(e))
            return {"error": str(e), "status": "error"}
