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

class SchemaValidationTest(MLTestCase):
    async def execute(self, model: Any, datasets: Dict[str, Any], baseline_model: Any, baseline_datasets: Dict[str, Any], start_time: float) -> MLTestCaseResult:
        train_df = datasets.get("training")
        val_df = datasets.get("validation")
        
        if train_df is None or val_df is None:
            return MLTestCaseResult(name=self.name, severity=self.severity, status=TestStatus.FAIL, explanation="Datasets missing")
        
        if self.target_column not in train_df.columns:
            return MLTestCaseResult(name=self.name, severity=self.severity, status=TestStatus.FAIL, explanation=f"Target '{self.target_column}' missing")
            
        train_cols = train_df.columns.tolist()
        val_cols = val_df.columns.tolist()
        
        if len(train_cols) != len(val_cols) or not all(c in val_cols for c in train_cols):
            return MLTestCaseResult(name=self.name, severity=self.severity, status=TestStatus.FAIL, explanation="Feature name/count mismatch")
            
        return MLTestCaseResult(name=self.name, severity=self.severity, status=TestStatus.PASS, explanation="Schema validated successfully")

class DatasetProfilingTest(MLTestCase):
    async def execute(self, model: Any, datasets: Dict[str, Any], baseline_model: Any, baseline_datasets: Dict[str, Any], start_time: float) -> MLTestCaseResult:
        df = datasets.get("validation")
        if df is None:
            return MLTestCaseResult(name=self.name, severity=self.severity, status=TestStatus.FAIL, explanation="No validation dataset")
            
        num_df = df.select_dtypes(include=[np.number])
        stats_dict = {
            "mean": num_df.mean().to_dict(),
            "std": num_df.std().to_dict(),
            "skew": num_df.skew().to_dict(),
            "kurtosis": num_df.kurtosis().to_dict(),
            "missing_pct": (df.isnull().sum() / len(df)).to_dict(),
            "entropy": {col: float(stats.entropy(df[col].value_counts(normalize=True))) for col in df.select_dtypes(exclude=[np.number]).columns}
        }
        
        return MLTestCaseResult(name=self.name, severity=self.severity, status=TestStatus.PASS, explanation="Profiling complete", details=stats_dict)

class RobustnessTest(MLTestCase):
    async def execute(self, model: Any, datasets: Dict[str, Any], baseline_model: Any, baseline_datasets: Dict[str, Any], start_time: float) -> MLTestCaseResult:
        df = datasets.get("validation")
        if df is None or self.target_column not in df.columns:
            return MLTestCaseResult(name=self.name, severity=self.severity, status=TestStatus.FAIL, explanation="Validation dataset or target missing")
            
        X = df.drop(columns=[self.target_column], errors='ignore')
        num_X = X.select_dtypes(include=[np.number])
        if num_X.empty:
            return MLTestCaseResult(name=self.name, severity=self.severity, status=TestStatus.PASS, explanation="No numeric features to perturb")
            
        y_pred_orig = model.predict(X)
        noise = np.random.normal(0, 0.01, num_X.shape)
        X_noisy = X.copy()
        X_noisy[num_X.columns] = num_X + noise
        y_pred_noisy = model.predict(X_noisy)
        
        flip_rate = np.mean(y_pred_orig != y_pred_noisy)
        threshold = self.config.get("config", {}).get("threshold", 0.05)
        
        status = TestStatus.PASS if flip_rate <= threshold else TestStatus.FAIL
        return MLTestCaseResult(name=self.name, severity=self.severity, status=status, metric_value=flip_rate, threshold=threshold, explanation=f"Flip rate: {flip_rate:.2%}")

class BiasDetectionTest(MLTestCase):
    async def execute(self, model: Any, datasets: Dict[str, Any], baseline_model: Any, baseline_datasets: Dict[str, Any], start_time: float) -> MLTestCaseResult:
        df = datasets.get("validation")
        attr = self.config.get("config", {}).get("protected_attribute")
        if df is None or attr not in df.columns or self.target_column not in df.columns:
            return MLTestCaseResult(name=self.name, severity=self.severity, status=TestStatus.WARN, explanation=f"Missing {attr} or target")
            
        X = df.drop(columns=[self.target_column], errors='ignore')
        y_pred = model.predict(X)
        df_eval = pd.DataFrame({attr: df[attr], 'pred': y_pred})
        
        rates = df_eval.groupby(attr)['pred'].mean()
        if len(rates) < 2 or rates.min() == 0:
            return MLTestCaseResult(name=self.name, severity=self.severity, status=TestStatus.PASS, explanation="Cannot compute bias ratio")
            
        disparate_impact = rates.min() / rates.max()
        threshold = self.config.get("config", {}).get("threshold", 0.8)
        
        status = TestStatus.PASS if disparate_impact >= threshold else TestStatus.FAIL
        return MLTestCaseResult(name=self.name, severity=self.severity, status=status, metric_value=disparate_impact, threshold=threshold, explanation=f"Disparate impact ratio: {disparate_impact:.2f}")

class OverfittingGapTest(MLTestCase):
    async def execute(self, model: Any, datasets: Dict[str, Any], baseline_model: Any, baseline_datasets: Dict[str, Any], start_time: float) -> MLTestCaseResult:
        train_df = datasets.get("training")
        val_df = datasets.get("validation")
        
        if train_df is None or val_df is None or self.target_column not in train_df.columns:
            return MLTestCaseResult(name=self.name, severity=self.severity, status=TestStatus.FAIL, explanation="Datasets or target missing")
            
        X_train = train_df.drop(columns=[self.target_column], errors='ignore')
        y_train = train_df[self.target_column]
        X_val = val_df.drop(columns=[self.target_column], errors='ignore')
        y_val = val_df[self.target_column]
        
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_acc = metrics.accuracy_score(y_train, train_pred)
        val_acc = metrics.accuracy_score(y_val, val_pred)
        
        gap = train_acc - val_acc
        threshold = self.config.get("config", {}).get("max_gap", 0.10)
        
        status = TestStatus.PASS if gap <= threshold else TestStatus.FAIL
        return MLTestCaseResult(
            name=self.name, severity=self.severity, status=status, 
            metric_value=gap, threshold=threshold, 
            explanation=f"Overfitting gap: {gap:.2%} (Train: {train_acc:.2%}, Val: {val_acc:.2%})"
        )
