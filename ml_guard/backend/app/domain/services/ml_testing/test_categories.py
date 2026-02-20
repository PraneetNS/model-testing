"""
ML Test Categories - Implementations for all 5 test categories
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from sklearn import metrics
import structlog

logger = structlog.get_logger(__name__)

class DataQualityTests:
    """Data quality validation tests."""

    def run_test(self, test_config: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Run a data quality test."""

        test_name = test_config.get("name", "Data Quality Test")
        test_type = test_config.get("type", "missing_values")

        if test_type == "missing_values":
            return self._test_missing_values(test_config, datasets)
        elif test_type == "duplicate_rows":
            return self._test_duplicate_rows(test_config, datasets)
        elif test_type == "outliers":
            return self._test_outliers(test_config, datasets)
        elif test_type == "schema_validation":
            return self._test_schema_validation(test_config, datasets)
        elif test_type == "class_balance":
            return self._test_class_balance(test_config, datasets)
        else:
            return {
                "status": "failed",
                "message": f"Unknown data quality test type: {test_type}"
            }

    def _test_missing_values(self, test_config: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Test for missing values in datasets."""

        threshold = test_config.get("config", {}).get("threshold", 0.05)
        dataset_name = test_config.get("config", {}).get("dataset", "validation")

        if dataset_name not in datasets:
            return {
                "status": "failed",
                "message": f"Dataset '{dataset_name}' not found"
            }

        df = datasets[dataset_name]
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_rate = missing_cells / total_cells if total_cells > 0 else 0

        # Check per-column missing rates
        column_missing_rates = df.isnull().mean()
        high_missing_columns = column_missing_rates[column_missing_rates > threshold]

        status = "passed" if missing_rate <= threshold else "failed"
        message = f"Missing value rate is {missing_rate:.2%} (Threshold: {threshold:.2%})"

        return {
            "status": status,
            "message": message,
            "details": {
                "total_missing_rate": missing_rate,
                "threshold": threshold,
                "missing_cells": int(missing_cells),
                "total_cells": total_cells,
                "high_missing_columns": high_missing_columns.to_dict() if len(high_missing_columns) > 0 else {}
            },
            "threshold": threshold,
            "actual_value": missing_rate
        }

    def _test_duplicate_rows(self, test_config: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Test for duplicate rows in datasets."""

        allow_duplicates = test_config.get("config", {}).get("allow_duplicates", False)
        dataset_name = test_config.get("config", {}).get("dataset", "validation")

        if dataset_name not in datasets:
            return {
                "status": "failed",
                "message": f"Dataset '{dataset_name}' not found"
            }

        df = datasets[dataset_name]
        duplicate_count = df.duplicated().sum()
        duplicate_rate = duplicate_count / len(df) if len(df) > 0 else 0

        has_duplicates = duplicate_count > 0

        if allow_duplicates:
            status = "passed"
            message = f"Found {duplicate_count} duplicate rows (duplicates allowed)"
        else:
            status = "failed" if has_duplicates else "passed"
            message = f"Found {duplicate_count} duplicate rows" + (" - duplicates not allowed" if has_duplicates else "")

        return {
            "status": status,
            "message": message,
            "details": {
                "duplicate_count": int(duplicate_count),
                "duplicate_rate": duplicate_rate,
                "total_rows": len(df),
                "allow_duplicates": allow_duplicates
            }
        }

    def _test_class_balance(self, test_config: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Test for class balance in classification datasets."""

        max_imbalance_ratio = test_config.get("config", {}).get("max_imbalance_ratio", 10.0)
        target_column = test_config.get("config", {}).get("target_column", "target")
        dataset_name = test_config.get("config", {}).get("dataset", "training")

        if dataset_name not in datasets:
            return {
                "status": "failed",
                "message": f"Dataset '{dataset_name}' not found"
            }

        df = datasets[dataset_name]

        if target_column not in df.columns:
            return {
                "status": "failed",
                "message": f"Target column '{target_column}' not found in dataset"
            }

        class_counts = df[target_column].value_counts()
        majority_class = class_counts.max()
        minority_class = class_counts.min()
        imbalance_ratio = majority_class / minority_class if minority_class > 0 else float('inf')

        status = "passed" if imbalance_ratio <= max_imbalance_ratio else "failed"
        message = f"Imbalance ratio: {imbalance_ratio:.2f} (Max Allowed: {max_imbalance_ratio:.2f})"

        return {
            "status": status,
            "message": message,
            "details": {
                "imbalance_ratio": imbalance_ratio,
                "max_allowed_ratio": max_imbalance_ratio,
                "class_counts": class_counts.to_dict(),
                "majority_class_count": int(majority_class),
                "minority_class_count": int(minority_class)
            },
            "threshold": max_imbalance_ratio,
            "actual_value": imbalance_ratio
        }

    def get_available_tests(self) -> List[str]:
        """Get list of available data quality tests."""
        return [
            "missing_values",
            "duplicate_rows",
            "outliers",
            "schema_validation",
            "class_balance"
        ]

    def validate_config(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality test configuration."""
        required_fields = ["category", "type"]
        for field in required_fields:
            if field not in test_config:
                return {"valid": False, "error": f"Missing required field: {field}"}

        if test_config["category"] != "data_quality":
            return {"valid": False, "error": "Invalid category for data quality test"}

        return {"valid": True}


class StatisticalStabilityTests:
    """Statistical stability and drift detection tests."""

    def run_test(self, test_config: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Run a statistical stability test."""

        test_type = test_config.get("type", "psi_drift")

        if test_type == "psi_drift":
            return self._test_psi_drift(test_config, datasets)
        elif test_type == "ks_test":
            return self._test_ks_drift(test_config, datasets)
        elif test_type == "correlation_stability":
            return self._test_correlation_stability(test_config, datasets)
        else:
            return {
                "status": "failed",
                "message": f"Unknown statistical stability test type: {test_type}"
            }

    def _test_psi_drift(self, test_config: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Test for population stability index drift between datasets."""

        psi_threshold = test_config.get("config", {}).get("psi_threshold", 0.1)
        feature_columns = test_config.get("config", {}).get("feature_columns", [])

        train_dataset = datasets.get("training")
        val_dataset = datasets.get("validation")

        if train_dataset is None or val_dataset is None:
            return {
                "status": "failed",
                "message": "Both training and validation datasets required for PSI test"
            }

        # If no specific columns provided, use all numeric columns
        if not feature_columns:
            numeric_cols = train_dataset.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_cols if col in val_dataset.columns]

        psi_scores = {}
        drifted_features = []

        for feature in feature_columns:
            if feature not in train_dataset.columns or feature not in val_dataset.columns:
                continue

            train_values = train_dataset[feature].dropna()
            val_values = val_dataset[feature].dropna()

            if len(train_values) == 0 or len(val_values) == 0:
                continue

            psi_score = self._calculate_psi(train_values, val_values)
            psi_scores[feature] = psi_score

            if psi_score > psi_threshold:
                drifted_features.append(feature)

        max_psi = max(psi_scores.values()) if psi_scores else 0
        status = "passed" if max_psi <= psi_threshold else "failed"
        message = f"Max PSI drift detected: {max_psi:.4f} (Threshold: {psi_threshold})"

        return {
            "status": status,
            "message": message,
            "details": {
                "psi_scores": psi_scores,
                "drifted_features": drifted_features,
                "max_psi_score": max_psi,
                "threshold": psi_threshold,
                "features_analyzed": len(feature_columns)
            },
            "threshold": psi_threshold,
            "actual_value": max_psi
        }

    def _calculate_psi(self, train_values: pd.Series, val_values: pd.Series, bins: int = 10) -> float:
        """
        Calculate Population Stability Index with mathematical rigor.
        Formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
        """
        try:
            # Add small epsilon to avoid divide by zero or log(0)
            eps = 1e-6
            
            # Use combined data to define consistent bins
            all_vals = pd.concat([train_values, val_values])
            if all_vals.nunique() <= bins:
                # For categorical or low-cardinality data, use unique values as bins
                train_counts = train_values.value_counts(normalize=True).sort_index()
                val_counts = val_values.value_counts(normalize=True).sort_index()
                
                # Align indices
                all_indices = train_counts.index.union(val_counts.index)
                train_pct = train_counts.reindex(all_indices, fill_value=0) + eps
                val_pct = val_counts.reindex(all_indices, fill_value=0) + eps
            else:
                # For continuous data, use quantile binning from training
                _, bin_edges = pd.qcut(train_values, q=bins, duplicates='drop', retbins=True)
                
                # Digitize both datasets into these bins
                train_hist, _ = np.histogram(train_values, bins=bin_edges)
                val_hist, _ = np.histogram(val_values, bins=bin_edges)
                
                train_pct = (train_hist / len(train_values)) + eps
                val_pct = (val_hist / len(val_values)) + eps

            # Calculate PSI using divergence formula
            psi_values = (val_pct - train_pct) * np.log(val_pct / train_pct)
            return float(np.sum(psi_values))

        except Exception as e:
            logger.error("PSI calculation failed", error=str(e))
            return 0.0

    def _test_ks_drift(self, test_config: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Test for Kolmogorov-Smirnov drift between datasets."""
        threshold = test_config.get("config", {}).get("p_value_threshold", 0.05)
        
        train_df = datasets.get("training")
        val_df = datasets.get("validation")
        
        if train_df is None or val_df is None:
            return {"status": "failed", "message": "Missing datasets for KS test"}
            
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        drifted_features = []
        p_values = {}
        
        for col in numeric_cols:
            if col in val_df.columns:
                stat, p_val = stats.ks_2samp(train_df[col].dropna(), val_df[col].dropna())
                p_values[col] = p_val
                if p_val < threshold:
                    drifted_features.append(col)
                    
        status = "passed" if len(drifted_features) == 0 else "failed"
        return {
            "status": status,
            "message": f"KS Test: {len(drifted_features)} features drifted" if status == "failed" else "No significant KS drift detected",
            "details": {"p_values": p_values, "drifted_features": drifted_features},
            "threshold": threshold,
            "actual_value": len(drifted_features)
        }

    def _test_correlation_stability(self, test_config: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Test if feature correlations remain stable between datasets."""
        threshold = test_config.get("config", {}).get("threshold", 0.9)
        
        train_df = datasets.get("training")
        val_df = datasets.get("validation")
        
        if train_df is None or val_df is None:
            return {"status": "failed", "message": "Missing datasets for correlation test"}
            
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.intersection(val_df.columns)
        if len(numeric_cols) < 2:
            return {"status": "passed", "message": "Insufficient numeric columns for correlation test"}
            
        corr_train = train_df[numeric_cols].corr()
        corr_val = val_df[numeric_cols].corr()
        
        # Calculate correlation of correlations
        corr_stability = corr_train.corrwith(corr_val).mean()
        
        status = "passed" if corr_stability >= threshold else "failed"
        return {
            "status": status,
            "message": f"Correlation Stability: {corr_stability:.3f} (Target: {threshold})",
            "details": {"stability_score": corr_stability},
            "threshold": threshold,
            "actual_value": corr_stability
        }

    def get_available_tests(self) -> List[str]:
        """Get list of available statistical stability tests."""
        return [
            "psi_drift",
            "ks_test",
            "correlation_stability"
        ]

    def validate_config(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate statistical stability test configuration."""
        required_fields = ["category", "type"]
        for field in required_fields:
            if field not in test_config:
                return {"valid": False, "error": f"Missing required field: {field}"}

        if test_config["category"] != "statistical_stability":
            return {"valid": False, "error": "Invalid category for statistical stability test"}

        return {"valid": True}


class ModelPerformanceTests:
    """Model performance validation tests."""

    def run_test(self, test_config: Dict[str, Any], model_artifact: Any, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Run a model performance test."""

        test_type = test_config.get("type", "accuracy_threshold")

        # Robust Prediction Handler
        self.model = model_artifact
        self.pipeline_meta = model_artifact if isinstance(model_artifact, dict) else None
        if self.pipeline_meta:
            self.model = self.pipeline_meta.get("model")

        if test_type == "accuracy_threshold":
            return self._test_accuracy_threshold(test_config, self.model, datasets)
        elif test_type == "precision_threshold":
            return self._test_precision_threshold(test_config, self.model, datasets)
        elif test_type == "recall_threshold":
            return self._test_recall_threshold(test_config, self.model, datasets)
        elif test_type == "f1_threshold":
            return self._test_f1_threshold(test_config, self.model, datasets)
        elif test_type == "roc_auc_threshold":
            return self._test_roc_auc_threshold(test_config, self.model, datasets)
        else:
            return {
                "status": "failed",
                "message": f"Unknown model performance test type: {test_type}"
            }

    def _get_X_y(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features and target, handling encoders if present."""
        if self.pipeline_meta and "features" in self.pipeline_meta:
            features = self.pipeline_meta["features"]
            # Filter columns that actually exist to avoid crash
            existing_features = [f for f in features if f in df.columns]
            X = df[existing_features]
        else:
            # Fallback: drop target and sensitive attributes if we don't have a feature list
            X = df.drop(columns=[target_column], errors='ignore')
        
        y = df[target_column]
        return X, y

    def _test_accuracy_threshold(self, test_config: Dict[str, Any], model: Any, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Test model accuracy against threshold."""

        threshold = test_config.get("config", {}).get("threshold", 0.85)
        operator = test_config.get("config", {}).get("operator", "gte")
        dataset_name = test_config.get("config", {}).get("dataset", "validation")
        target_column = test_config.get("config", {}).get("target_column", "target")

        if dataset_name not in datasets:
            return {
                "status": "failed",
                "message": f"Dataset '{dataset_name}' not found"
            }

        df = datasets[dataset_name]

        if target_column not in df.columns:
            return {
                "status": "failed",
                "message": f"Target column '{target_column}' not found"
            }

        # Get features and target using alignment helper
        X, y_true = self._get_X_y(df, target_column)

        # Make predictions
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)
                if len(np.unique(y_true)) == 2:  # Binary classification
                    y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
                else:  # Multi-class
                    y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(X)

            accuracy = metrics.accuracy_score(y_true, y_pred)

            # Apply operator
            if operator == "gte":
                passed = accuracy >= threshold
            elif operator == "gt":
                passed = accuracy > threshold
            elif operator == "lte":
                passed = accuracy <= threshold
            elif operator == "lt":
                passed = accuracy < threshold
            else:
                return {
                    "status": "failed",
                    "message": f"Unknown operator: {operator}"
                }

            status = "passed" if passed else "failed"
            message = f"Model Accuracy: {accuracy:.4f} (Required: {threshold:.4f})"

            return {
                "status": status,
                "message": message,
                "details": {
                    "accuracy": accuracy,
                    "threshold": threshold,
                    "operator": operator,
                    "dataset_size": len(df)
                },
                "threshold": threshold,
                "actual_value": accuracy
            }

        except Exception as e:
            return {
                "status": "failed",
                "message": f"Model prediction failed: {str(e)}"
            }

    def _test_precision_threshold(self, test_config: Dict[str, Any], model: Any, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Test model precision against threshold."""
        return self._test_metric_threshold(test_config, model, datasets, "precision")

    def _test_recall_threshold(self, test_config: Dict[str, Any], model: Any, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Test model recall against threshold."""
        return self._test_metric_threshold(test_config, model, datasets, "recall")

    def _test_f1_threshold(self, test_config: Dict[str, Any], model: Any, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Test model F1 score against threshold."""
        return self._test_metric_threshold(test_config, model, datasets, "f1")

    def _test_metric_threshold(self, test_config: Dict[str, Any], model: Any, datasets: Dict[str, Any], metric: str) -> Dict[str, Any]:
        """Test a specific metric against threshold."""

        threshold = test_config.get("config", {}).get("threshold", 0.8)
        operator = test_config.get("config", {}).get("operator", "gte")
        dataset_name = test_config.get("config", {}).get("dataset", "validation")
        target_column = test_config.get("config", {}).get("target_column", "target")

        if dataset_name not in datasets:
            return {
                "status": "failed",
                "message": f"Dataset '{dataset_name}' not found"
            }

        df = datasets[dataset_name]

        if target_column not in df.columns:
            return {
                "status": "failed",
                "message": f"Target column '{target_column}' not found"
            }

        # Get features (exclude target)
        # Get features and target using alignment helper
        X, y_true = self._get_X_y(df, target_column)

        # Make predictions
        try:
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)
                if len(np.unique(y_true)) == 2:  # Binary classification
                    y_pred = (pred_proba[:, 1] > 0.5).astype(int)
                else:  # Multi-class
                    y_pred = np.argmax(pred_proba, axis=1)
            else:
                y_pred = model.predict(X)

            # Calculate the requested metric
            if metric == "precision":
                value = metrics.precision_score(y_true, y_pred, average='weighted')
                metric_name = "Precision"
            elif metric == "recall":
                value = metrics.recall_score(y_true, y_pred, average='weighted')
                metric_name = "Recall"
            elif metric == "f1":
                value = metrics.f1_score(y_true, y_pred, average='weighted')
                metric_name = "F1 Score"
            else:
                return {
                    "status": "failed",
                    "message": f"Unknown metric: {metric}"
                }

            # Apply operator
            if operator == "gte":
                passed = value >= threshold
            elif operator == "gt":
                passed = value > threshold
            elif operator == "lte":
                passed = value <= threshold
            elif operator == "lt":
                passed = value < threshold
            else:
                return {
                    "status": "failed",
                    "message": f"Unknown operator: {operator}"
                }

            status = "passed" if passed else "failed"
            message = f"{metric_name} {value:.3f} {'≥' if operator == 'gte' else '>' if operator == 'gt' else '≤' if operator == 'lte' else '<'} {threshold}"

            return {
                "status": status,
                "message": message,
                "details": {
                    f"{metric}": value,
                    "threshold": threshold,
                    "operator": operator,
                    "dataset_size": len(df)
                },
                "threshold": threshold,
                "actual_value": value
            }

        except Exception as e:
            return {
                "status": "failed",
                "message": f"Model prediction failed: {str(e)}"
            }

    def _test_input_perturbation(self, test_config: Dict[str, Any], model: Any, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Test model sensitivity to input perturbations."""

        perturbation_factor = test_config.get("config", {}).get("perturbation_factor", 0.1)
        dataset_name = test_config.get("config", {}).get("dataset", "validation")
        sensitivity_threshold = test_config.get("config", {}).get("sensitivity_threshold", 0.1)

        if dataset_name not in datasets:
            return {
                "status": "failed",
                "message": f"Dataset '{dataset_name}' not found"
            }

        df = datasets[dataset_name]
        target_column = test_config.get("config", {}).get("target_column", "churn")
        feature_cols = [col for col in df.columns if col != target_column]

        if not feature_cols:
            return {
                "status": "failed",
                "message": "No feature columns found"
            }

        # Sample a subset for efficiency
        sample_size = min(100, len(df))
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
        X_sample = df.iloc[sample_indices][feature_cols].values

        # Make predictions on original data
        try:
            if hasattr(model, 'predict_proba'):
                original_preds = model.predict_proba(X_sample)
                if original_preds.shape[1] == 2:  # Binary classification
                    original_confidence = np.max(original_preds, axis=1)
                else:
                    original_confidence = np.max(original_preds, axis=1)
            else:
                original_preds = model.predict(X_sample)
                original_confidence = np.ones(len(original_preds))  # Assume confidence of 1 for non-probabilistic models

            # Apply perturbations and measure sensitivity
            sensitivity_scores = []

            for i in range(sample_size):
                # Create perturbed version of this sample
                perturbation = np.random.normal(0, perturbation_factor, X_sample[i].shape)
                perturbed_sample = X_sample[i] + perturbation

                # Make prediction on perturbed data
                try:
                    if hasattr(model, 'predict_proba'):
                        perturbed_preds = model.predict_proba(perturbed_sample.reshape(1, -1))
                        if perturbed_preds.shape[1] == 2:  # Binary classification
                            perturbed_confidence = np.max(perturbed_preds, axis=1)[0]
                        else:
                            perturbed_confidence = np.max(perturbed_preds, axis=1)[0]
                    else:
                        perturbed_pred = model.predict(perturbed_sample.reshape(1, -1))
                        perturbed_confidence = 1.0  # Assume confidence of 1

                    # Calculate sensitivity (change in confidence)
                    sensitivity = abs(original_confidence[i] - perturbed_confidence)
                    sensitivity_scores.append(sensitivity)

                except Exception:
                    continue

            if not sensitivity_scores:
                return {
                    "status": "failed",
                    "message": "Could not calculate perturbation sensitivity"
                }

            avg_sensitivity = np.mean(sensitivity_scores)
            max_sensitivity = np.max(sensitivity_scores)

            # Check if sensitivity is within acceptable range
            status = "passed" if avg_sensitivity <= sensitivity_threshold else "failed"

            message = ".3f" if status == "passed" else ".3f"

            return {
                "status": status,
                "message": message,
                "details": {
                    "average_sensitivity": float(avg_sensitivity),
                    "max_sensitivity": float(max_sensitivity),
                    "perturbation_factor": perturbation_factor,
                    "sample_size": sample_size,
                    "threshold": sensitivity_threshold
                },
                "threshold": sensitivity_threshold,
                "actual_value": avg_sensitivity
            }

        except Exception as e:
            return {
                "status": "failed",
                "message": f"Input perturbation test failed: {str(e)}"
            }

    def get_available_tests(self) -> List[str]:
        """Get list of available model performance tests."""
        return [
            "accuracy_threshold",
            "precision_threshold",
            "recall_threshold",
            "f1_threshold",
            "roc_auc_threshold"
        ]

    def validate_config(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model performance test configuration."""
        required_fields = ["category", "type"]
        for field in required_fields:
            if field not in test_config:
                return {"valid": False, "error": f"Missing required field: {field}"}

        if test_config["category"] != "model_performance":
            return {"valid": False, "error": "Invalid category for model performance test"}

        return {"valid": True}


class RobustnessTests:
    """Model robustness and stability tests."""

    def run_test(self, test_config: Dict[str, Any], model_artifact: Any, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Run a robustness test."""

        test_type = test_config.get("type", "prediction_stability")

        if test_type == "prediction_stability":
            return self._test_prediction_stability(test_config, model_artifact, datasets)
        elif test_type == "input_perturbation":
            return self._test_input_perturbation(test_config, model_artifact, datasets)
        else:
            return {
                "status": "failed",
                "message": f"Unknown robustness test type: {test_type}"
            }

    def _test_prediction_stability(self, test_config: Dict[str, Any], model: Any, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Test prediction stability with slight input variations."""

        stability_threshold = test_config.get("config", {}).get("stability_threshold", 0.95)
        noise_level = test_config.get("config", {}).get("noise_level", 0.01)
        dataset_name = test_config.get("config", {}).get("dataset", "validation")

        if dataset_name not in datasets:
            return {
                "status": "failed",
                "message": f"Dataset '{dataset_name}' not found"
            }

        df = datasets[dataset_name]
        target_column = test_config.get("config", {}).get("target_column", "target")
        feature_cols = [col for col in df.columns if col != target_column]

        if not feature_cols:
            return {
                "status": "failed",
                "message": "No feature columns found"
            }

        X = df[feature_cols].values
        sample_size = min(100, len(X))  # Test on subset for performance

        stable_predictions = 0
        total_predictions = 0

        for i in range(sample_size):
            # Get original prediction
            original_sample = X[i:i+1]
            try:
                if hasattr(model, 'predict_proba'):
                    original_pred = model.predict_proba(original_sample)[0]
                else:
                    original_pred = model.predict(original_sample)

                # Add small noise and get new prediction
                noise = np.random.normal(0, noise_level, original_sample.shape)
                noisy_sample = original_sample + noise

                if hasattr(model, 'predict_proba'):
                    noisy_pred = model.predict_proba(noisy_sample)[0]
                else:
                    noisy_pred = model.predict(noisy_sample)

                # Check if predictions are stable (same class for classification)
                if np.array_equal(np.argmax(original_pred), np.argmax(noisy_pred)):
                    stable_predictions += 1

                total_predictions += 1

            except Exception:
                continue

        stability_rate = stable_predictions / total_predictions if total_predictions > 0 else 0

        status = "passed" if stability_rate >= stability_threshold else "failed"
        message = f"Prediction stability: {stability_rate:.1%} (Target: {stability_threshold:.1%})"

        return {
            "status": status,
            "message": message,
            "details": {
                "stability_rate": stability_rate,
                "threshold": stability_threshold,
                "stable_predictions": stable_predictions,
                "total_predictions": total_predictions,
                "noise_level": noise_level,
                "sample_size": sample_size
            },
            "threshold": stability_threshold,
            "actual_value": stability_rate
        }

    def get_available_tests(self) -> List[str]:
        """Get list of available robustness tests."""
        return [
            "prediction_stability",
            "input_perturbation"
        ]

    def validate_config(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate robustness test configuration."""
        required_fields = ["category", "type"]
        for field in required_fields:
            if field not in test_config:
                return {"valid": False, "error": f"Missing required field: {field}"}

        if test_config["category"] != "robustness":
            return {"valid": False, "error": "Invalid category for robustness test"}

        return {"valid": True}


class BiasFairnessTests:
    """Bias detection and fairness tests."""

    def run_test(self, test_config: Dict[str, Any], model_artifact: Any, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Run a bias and fairness test."""

        test_type = test_config.get("type", "disparate_impact")

        if test_type == "disparate_impact":
            return self._test_disparate_impact(test_config, model_artifact, datasets)
        elif test_type == "equal_opportunity":
            return self._test_equal_opportunity(test_config, model_artifact, datasets)
        else:
            return {
                "status": "failed",
                "message": f"Unknown bias/fairness test type: {test_type}"
            }

    def _test_disparate_impact(self, test_config: Dict[str, Any], model: Any, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Test for disparate impact bias."""

        threshold = test_config.get("config", {}).get("threshold", 1.2)
        protected_attribute = test_config.get("config", {}).get("protected_attribute", "gender")
        dataset_name = test_config.get("config", {}).get("dataset", "validation")
        target_column = test_config.get("config", {}).get("target_column", "target")

        if dataset_name not in datasets:
            return {
                "status": "failed",
                "message": f"Dataset '{dataset_name}' not found"
            }

        df = datasets[dataset_name]

        if protected_attribute not in df.columns:
            return {
                "status": "failed",
                "message": f"Protected attribute '{protected_attribute}' not found"
            }

        if target_column not in df.columns:
            return {
                "status": "failed",
                "message": f"Target column '{target_column}' not found"
            }

        # Get predictions
        # Get features and target using alignment helper
        X, _ = self._get_X_y(df, target_column)
        
        # Ensure protected attribute isn't in X if it was excluded during training
        if protected_attribute in X.columns:
            X = X.drop(columns=[protected_attribute])

        try:
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)
                if pred_proba.shape[1] == 2:  # Binary classification
                    predictions = (pred_proba[:, 1] > 0.5).astype(int)
                else:
                    predictions = np.argmax(pred_proba, axis=1)
            else:
                predictions = model.predict(X)

            # Calculate selection rates by protected attribute groups
            protected_values = df[protected_attribute].unique()

            if len(protected_values) < 2:
                return {
                    "status": "failed",
                    "message": f"Protected attribute '{protected_attribute}' must have at least 2 groups"
                }

            selection_rates = {}
            for value in protected_values:
                group_mask = df[protected_attribute] == value
                group_predictions = predictions[group_mask]
                selection_rate = np.mean(group_predictions) if len(group_predictions) > 0 else 0
                selection_rates[str(value)] = selection_rate

            # Calculate disparate impact ratio (max/min selection rate)
            rates = list(selection_rates.values())
            max_rate = max(rates)
            min_rate = min(rates)
            disparate_impact = max_rate / min_rate if min_rate > 0 else float('inf')

            status = "passed" if disparate_impact <= threshold else "failed"
            message = f"Disparate Impact Ratio: {disparate_impact:.3f} (Threshold: {threshold})"

            return {
                "status": status,
                "message": message,
                "details": {
                    "disparate_impact_ratio": disparate_impact,
                    "threshold": threshold,
                    "selection_rates": selection_rates,
                    "protected_attribute": protected_attribute,
                    "protected_groups": len(protected_values)
                },
                "threshold": threshold,
                "actual_value": disparate_impact
            }
        except Exception as e:
            return {"status": "failed", "message": f"Disparate impact test failed: {str(e)}"}

    def _test_equal_opportunity(self, test_config: Dict[str, Any], model: Any, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test for Equal Opportunity Difference bias.
        Formula: |TPR_group_A - TPR_group_B| <= threshold
        """
        threshold = test_config.get("config", {}).get("threshold", 0.1)
        protected_attribute = test_config.get("config", {}).get("protected_attribute", "gender")
        dataset_name = test_config.get("config", {}).get("dataset", "validation")
        target_column = test_config.get("config", {}).get("target_column", "target")

        if dataset_name not in datasets:
            return {"status": "failed", "message": f"Dataset '{dataset_name}' not found"}

        df = datasets[dataset_name]
        
        # Get features and target using alignment helper
        X, y_true = self._get_X_y(df, target_column)
        if protected_attribute in X.columns:
            X = X.drop(columns=[protected_attribute])

        try:
            # Get predictions
            predictions = model.predict(X)
            
            # Calculate TPR (True Positive Rate) for each group
            protected_values = df[protected_attribute].unique()
            tpr_by_group = {}
            
            for value in protected_values:
                group_mask = df[protected_attribute] == value
                group_y_true = y_true[group_mask]
                group_preds = predictions[group_mask]
                
                # TPR = TP / (TP + FN) = TP / Total Positives
                positives_mask = group_y_true == 1
                if positives_mask.sum() == 0:
                    tpr = 1.0 # Or 0? Usually 1 if no ground truth positives
                else:
                    tpr = np.mean(group_preds[positives_mask])
                tpr_by_group[str(value)] = float(tpr)

            # Max difference between any two groups
            tprs = list(tpr_by_group.values())
            max_diff = max(tprs) - min(tprs)
            
            status = "passed" if max_diff <= threshold else "failed"
            message = f"Equal Opportunity Difference: {max_diff:.3f} (Threshold: {threshold})"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "max_difference": max_diff,
                    "tpr_by_group": tpr_by_group,
                    "protected_attribute": protected_attribute
                },
                "actual_value": max_diff,
                "threshold": threshold
            }
        except Exception as e:
            return {"status": "failed", "message": f"Fairness test failed: {str(e)}"}

    def get_available_tests(self) -> List[str]:
        """Get list of available bias and fairness tests."""
        return [
            "disparate_impact",
            "equal_opportunity"
        ]

    def validate_config(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate bias and fairness test configuration."""
        required_fields = ["category", "type"]
        for field in required_fields:
            if field not in test_config:
                return {"valid": False, "error": f"Missing required field: {field}"}

        if test_config["category"] != "bias_fairness":
            return {"valid": False, "error": "Invalid category for bias/fairness test"}

        return {"valid": True}