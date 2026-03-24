import pandas as pd
import joblib
import io
import os
import structlog
from typing import Dict, Any, Optional, List
from fastapi import HTTPException
import numpy as np
from sklearn.base import is_classifier, is_regressor

logger = structlog.get_logger(__name__)

class ArtifactInspector:
    """
    Advanced Artifact Intelligence Engine.
    Performs deep profiling of datasets and neuro-architecture inspection of models.
    """
    
    @staticmethod
    def validate_extension(filename: str, allowed: list[str]) -> bool:
        ext = os.path.splitext(filename)[1].lower()
        return ext in allowed

    async def inspect_model(self, model_content: bytes, filename: str) -> Dict[str, Any]:
        if not self.validate_extension(filename, ['.pkl', '.joblib']):
            raise ValueError(f"Security Rejection: Invalid model format '{filename}'. Only .pkl and .joblib are allowed.")

        try:
            model = joblib.load(io.BytesIO(model_content))
            
            model_type = "unknown"
            try:
                # In sklearn 1.6+, is_classifier attempts to read __sklearn_tags__, which crashes for dict/primitives.
                if is_classifier(model) or getattr(model, "_estimator_type", None) == "classifier":
                    model_type = "classification"
                elif is_regressor(model) or getattr(model, "_estimator_type", None) == "regressor":
                    model_type = "regression"
            except (AttributeError, TypeError, ValueError, Exception):
                pass


            features_expected = None
            if hasattr(model, "n_features_in_"):
                features_expected = int(model.n_features_in_)
            elif hasattr(model, "feature_names_in_"):
                features_expected = len(model.feature_names_in_)

            classes = None
            class_count = 0
            if hasattr(model, "classes_"):
                classes = [str(c) for c in model.classes_]
                class_count = len(classes)

            return {
                "name": filename,
                "type": model_type,
                "subtype": "binary" if class_count == 2 else "multiclass" if class_count > 2 else "continuous",
                "features_expected": features_expected,
                "classes": classes,
                "status": "valid"
            }
        except Exception as e:
            logger.error("Model telemetry failed", error=str(e), filename=filename)
            raise ValueError(f"Artifact Extraction Failed: {str(e)}")

    async def profile_dataset(self, content: bytes, filename: str, target_column: Optional[str] = None) -> Dict[str, Any]:
        if not self.validate_extension(filename, ['.csv']):
            raise ValueError(f"Security Rejection: Invalid dataset format '{filename}'. Only .csv allowed.")

        try:
            df = pd.read_csv(io.BytesIO(content))
            if df.empty:
                raise ValueError("Payload Error: Dataset is empty.")

            rows, cols = df.shape
            mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024) # MB
            
            missing_pct = (df.isnull().sum().sum() / (rows * cols)) * 100
            duplicate_pct = (df.duplicated().sum() / rows) * 100
            
            dtypes = df.dtypes.apply(lambda x: str(x)).to_dict()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            # Intelligent Target Detection (Low cardinality columns or columns named 'target', 'label', 'churn')
            potential_targets = []
            for col in df.columns:
                unique_pct = df[col].nunique() / rows
                if col.lower() in ['target', 'label', 'churn', 'y', 'outcome'] or (df[col].nunique() < 20 and unique_pct < 0.05):
                    potential_targets.append(col)

            return {
                "filename": filename,
                "rows": rows,
                "columns": cols,
                "column_names": df.columns.tolist(),
                "dtypes": dtypes,
                "missing_percent": round(float(missing_pct), 2),
                "duplicate_percent": round(float(duplicate_pct), 2),
                "memory_mb": round(float(mem_usage), 2),
                "numeric_count": len(numeric_cols),
                "categorical_count": len(categorical_cols),
                "potential_targets": potential_targets[:5],
                "target_exists": target_column in df.columns if target_column else False,
                "status": "profiled"
            }
        except Exception as e:
            logger.error("Dataset profiling failed", error=str(e), filename=filename)
            raise ValueError(f"Data Profiling Error: {str(e)}")

    def validate_compatibility(self, model_meta: Dict, train_meta: Dict) -> List[str]:
        errors = []
        
        # 1. Feature Count Check
        if model_meta.get("features_expected"):
            # If target is in train_meta, subtract 1 from columns
            actual_features = train_meta["columns"] - (1 if train_meta.get("target_exists") else 0)
            if actual_features != model_meta["features_expected"]:
                errors.append(f"Feature Dimension Mismatch: Model expects {model_meta['features_expected']} features, but dataset provides {actual_features} (excluding target).")

        # 2. Target Check
        if not train_meta.get("target_exists"):
            errors.append("Structural Error: Target column not detected in training dataset.")

        return errors
