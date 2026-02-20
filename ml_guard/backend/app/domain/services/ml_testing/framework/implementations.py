import pandas as pd
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from sklearn import metrics
from .base import MLTestCase, MLTestCaseResult, TestStatus, Severity

class MissingValuesTest(MLTestCase):
    async def execute(self, model: Any, datasets: Dict[str, Any], baseline_model: Any, baseline_datasets: Dict[str, Any], start_time: float) -> MLTestCaseResult:
        threshold = self.config.get("config", {}).get("threshold", 0.05)
        dataset_name = self.config.get("config", {}).get("dataset", "validation")
        
        if dataset_name not in datasets:
            return MLTestCaseResult(
                name=self.name, description=self.description, severity=self.severity,
                status=TestStatus.ERROR, explanation=f"Dataset '{dataset_name}' not found",
                remediation="Ensure the required dataset is uploaded.", execution_time=time.time() - start_time
            )

        df = datasets[dataset_name]
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_rate = missing_cells / total_cells if total_cells > 0 else 0
        
        status = TestStatus.PASS if missing_rate <= threshold else TestStatus.FAIL
        return MLTestCaseResult(
            name=self.name, description=self.description, severity=self.severity, status=status,
            metric_value=missing_rate, threshold=threshold,
            explanation=f"Missing rate is {missing_rate:.2%}. Threshold is {threshold:.2%}.",
            remediation="Perform data imputation or check upstream data pipelines." if status == TestStatus.FAIL else "None",
            execution_time=time.time() - start_time,
            details={"missing_cells": int(missing_cells), "total_cells": total_cells}
        )

class AccuracyTest(MLTestCase):
    async def execute(self, model: Any, datasets: Dict[str, Any], baseline_model: Any, baseline_datasets: Dict[str, Any], start_time: float) -> MLTestCaseResult:
        threshold = self.config.get("config", {}).get("threshold", 0.80)
        dataset_name = self.config.get("config", {}).get("dataset", "validation")
        
        if dataset_name not in datasets:
            return MLTestCaseResult(
                name=self.name, description=self.description, severity=self.severity,
                status=TestStatus.ERROR, explanation=f"Dataset '{dataset_name}' not found",
                remediation="Check data availability.", execution_time=time.time() - start_time
            )

        df = datasets[dataset_name]
        if self.target_column not in df.columns:
            return MLTestCaseResult(
                name=self.name, description=self.description, severity=self.severity,
                status=TestStatus.ERROR, explanation=f"Target '{self.target_column}' missing",
                remediation="Verify target column name.", execution_time=time.time() - start_time
            )

        X = df.drop(columns=[self.target_column], errors='ignore')
        y_true = df[self.target_column]
        y_pred = model.predict(X)
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        status = TestStatus.PASS if accuracy >= threshold else TestStatus.FAIL
        
        return MLTestCaseResult(
            name=self.name, description=self.description, severity=self.severity, status=status,
            metric_value=accuracy, threshold=threshold,
            explanation=f"Model accuracy is {accuracy:.2%} (Threshold: {threshold:.2%}).",
            remediation="Retrain model with more data or better features." if status == TestStatus.FAIL else "None",
            execution_time=time.time() - start_time
        )

class PSIDriftTest(MLTestCase):
    async def execute(self, model: Any, datasets: Dict[str, Any], baseline_model: Any, baseline_datasets: Dict[str, Any], start_time: float) -> MLTestCaseResult:
        threshold = self.config.get("config", {}).get("psi_threshold", 0.1)
        train_df = datasets.get("training")
        val_df = datasets.get("validation")

        if train_df is None or val_df is None:
            return MLTestCaseResult(
                name=self.name, description=self.description, severity=self.severity,
                status=TestStatus.ERROR, explanation="Both training and validation required for PSI",
                remediation="Upload both datasets.", execution_time=time.time() - start_time
            )

        # Simple logic for now (max PSI across numeric features)
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        max_psi = 0
        drifted_col = None
        
        for col in numeric_cols[:10]: # Limit for perf
            if col in val_df.columns:
                psi = self._calculate_psi(train_df[col], val_df[col])
                if psi > max_psi:
                    max_psi = psi
                    drifted_col = col

        status = TestStatus.PASS if max_psi <= threshold else TestStatus.FAIL
        return MLTestCaseResult(
            name=self.name, description=self.description, severity=self.severity, status=status,
            metric_value=max_psi, threshold=threshold,
            explanation=f"Max PSI is {max_psi:.4f} (at {drifted_col})." if drifted_col else "No numeric features found.",
            remediation="Refresh the training data to match production distribution." if status == TestStatus.FAIL else "None",
            execution_time=time.time() - start_time
        )

    def _calculate_psi(self, expected, actual, bins=10):
        try:
            expected_percents, bin_edges = np.histogram(expected, bins=bins, density=False)
            actual_percents, _ = np.histogram(actual, bins=bin_edges, density=False)
            
            expected_percents = expected_percents / len(expected) + 1e-6
            actual_percents = actual_percents / len(actual) + 1e-6
            
            psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
            return float(psi)
        except: return 0.0

class RegressionTest(MLTestCase):
    async def execute(self, model: Any, datasets: Dict[str, Any], baseline_model: Any, baseline_datasets: Dict[str, Any], start_time: float) -> MLTestCaseResult:
        if baseline_model is None:
             return MLTestCaseResult(
                name=self.name, description=self.description, severity=self.severity,
                status=TestStatus.WARN, explanation="No baseline model provided for regression check.",
                remediation="Provide a baseline model artifact.", execution_time=time.time() - start_time
            )

        val_df = datasets.get("validation")
        X = val_df.drop(columns=[self.target_column], errors='ignore')
        y_true = val_df[self.target_column]
        
        current_acc = metrics.accuracy_score(y_true, model.predict(X))
        baseline_acc = metrics.accuracy_score(y_true, baseline_model.predict(X))
        
        diff = current_acc - baseline_acc
        threshold = self.config.get("config", {}).get("max_drop", -0.05)
        
        status = TestStatus.PASS if diff >= threshold else TestStatus.FAIL
        return MLTestCaseResult(
            name=self.name, description=self.description, severity=self.severity, status=status,
            metric_value=diff, threshold=threshold,
            explanation=f"Accuracy change: {diff:.2%}. Baseline: {baseline_acc:.2%}. Current: {current_acc:.2%}.",
            remediation="Investigate why the new model is performing worse than the previous version." if status == TestStatus.FAIL else "None",
            execution_time=time.time() - start_time
        )
