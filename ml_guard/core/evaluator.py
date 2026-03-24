import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .exceptions import ModelValidationError, DataMismatchError, SchemaError
from .constraints import Constraint, PredictorValidationRule
from .drift import compute_psi, compute_ks

class MLEvaluator:
    def __init__(self, model: Any, X_train: pd.DataFrame, y_train: Any, X_val: pd.DataFrame, y_val: Any):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.constraints: List[Constraint] = []
        self.rules: List[PredictorValidationRule] = []
        self.max_drift_threshold = 0.25
        self.max_overfitting_gap = 0.10
        self._validate_schema()
        
    def _validate_schema(self):
        if not list(self.X_train.columns) == list(self.X_val.columns):
            raise SchemaError("Train and validation sets must have exactly the same columns.")
        if len(self.X_train) != len(self.y_train):
            raise DataMismatchError("X_train and y_train lengths must match.")
        if len(self.X_val) != len(self.y_val):
            raise DataMismatchError("X_val and y_val lengths must match.")

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)
        
    def add_rule(self, rule: PredictorValidationRule):
        self.rules.append(rule)
        
    def set_max_drift_threshold(self, threshold: float):
        self.max_drift_threshold = threshold
        
    def set_max_overfitting_gap(self, gap: float):
        self.max_overfitting_gap = gap

    def evaluate(self) -> dict:
        results = {
            "status": "PASSED",
            "metrics": {},
            "drift": {},
            "violations": [],
            "overfitting_gap": {},
            "governance_score": 100,
            "critical_failures": []
        }
        
        # 1. Prediction Generation
        try:
            y_train_pred = self.model.predict(self.X_train)
            y_val_pred = self.model.predict(self.X_val)
            y_train_prob = self.model.predict_proba(self.X_train) if hasattr(self.model, "predict_proba") else None
            y_val_prob = self.model.predict_proba(self.X_val) if hasattr(self.model, "predict_proba") else None
        except Exception as e:
            raise ModelValidationError(f"Model prediction step failed: {str(e)}")
            
        # 2. Constraints & Overfitting
        for c in self.constraints:
            try:
                # Validation evaluation
                eval_res = c.evaluate(self.y_val, y_val_pred, y_val_prob)
                results["metrics"][c.name] = eval_res["actual_value"]
                
                if not eval_res["passed"]:
                    results["violations"].append(eval_res)
                    
                # Train evaluation to compute overfitting gap
                train_val = c.metric_function(self.y_train, y_train_pred, y_train_prob)
                gap = train_val - eval_res["actual_value"]
                results["overfitting_gap"][c.name] = gap
                
                if gap > self.max_overfitting_gap:
                    results["violations"].append({
                        "name": f"Overfitting: {c.name}",
                        "passed": False,
                        "actual_gap": gap,
                        "threshold": self.max_overfitting_gap,
                        "reason": f"Overfitting gap for {c.name} ({gap:.4f}) exceeds threshold ({self.max_overfitting_gap})."
                    })
                    
            except Exception as e:
                results["critical_failures"].append(f"Constraint '{c.name}' computation failed: {str(e)}")

        # 3. Custom Rules
        for r in self.rules:
            try:
                # Need to use a df with reset index for iteration safety
                rule_res = r.evaluate(self.X_val.reset_index(drop=True), y_val_pred)
                if not rule_res["passed"]:
                    results["violations"].append(rule_res)
            except Exception as e:
                results["critical_failures"].append(f"Custom rule '{r.name}' computation failed: {str(e)}")
                
        # 4. Drift Analysis
        numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                psi = compute_psi(self.X_train[col].values, self.X_val[col].values)
                ks_stat, _ = compute_ks(self.X_train[col].values, self.X_val[col].values)
                results["drift"][col] = {
                    "PSI": psi,
                    "KS_Stat": ks_stat
                }
                if psi > self.max_drift_threshold:
                    results["violations"].append({
                        "name": f"Drift: {col}",
                        "passed": False,
                        "actual_psi": psi,
                        "threshold": self.max_drift_threshold,
                        "reason": f"Feature '{col}' PSI ({psi:.4f}) exceeds drift threshold ({self.max_drift_threshold})."
                    })
            except Exception as e:
                 results["critical_failures"].append(f"Drift computation for '{col}' failed: {str(e)}")
                 
        # 5. Governance Scoring (Deterministic Penalty)
        score = 100
        violation_types = [v["name"] for v in results["violations"]]
        score -= len([v for v in violation_types if not v.startswith("Drift:")]) * 10
        score -= len([v for v in violation_types if v.startswith("Drift:")]) * 5
        score -= len(results["critical_failures"]) * 50
        
        results["governance_score"] = max(0, score)
        
        if len(results["violations"]) > 0 or len(results["critical_failures"]) > 0:
            results["status"] = "FAILED"
            
        return results
