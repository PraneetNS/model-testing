
import pandas as pd
import joblib
import uuid
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import structlog

logger = structlog.get_logger(__name__)

class Trainer:
    """
    Production-grade model trainer with automated preprocessing, 
    multiple algorithm support, and comprehensive evaluation.
    """
    def __init__(self, storage_path: str = "storage/models"):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        self.encoders = {}

    def _preprocess_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Handles categorical encoding, missing values, and feature selection.
        """
        df = df.copy()
        
        # 1. Handle Missing Values
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                # For categorical/string, use mode
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "UNKNOWN")

        # 2. Separate Features and Target
        y = df[target_column]
        X = df.drop(columns=[target_column])

        # Encode target if it's categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            target_le = LabelEncoder()
            y = target_le.fit_transform(y.astype(str))
            self.encoders[target_column] = target_le

        # 3. Categorical Feature Encoding (Label Encoding for simplicity, could be OneHot)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le

        return X, y, list(X.columns)

    def train_model(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        model_type: str = "random_forest",
        hyperparams: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2,
        do_cv: bool = True
    ) -> Dict[str, Any]:
        """
        Full training pipeline with evaluation and persistence.
        """
        logger.info("Starting production training pipeline", model_type=model_type, target=target_column)
        
        try:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data.")

            # Preprocessing
            X, y, features = self._preprocess_data(df, target_column)
            
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Model Selection
            params = hyperparams or {}
            if model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=params.get("n_estimators", 100),
                    max_depth=params.get("max_depth", None),
                    min_samples_split=params.get("min_samples_split", 2),
                    random_state=42
                )
            elif model_type == "gradient_boosting":
                model = GradientBoostingClassifier(
                    n_estimators=params.get("n_estimators", 100),
                    learning_rate=params.get("learning_rate", 0.1),
                    random_state=42
                )
            elif model_type == "logistic_regression":
                model = LogisticRegression(max_iter=1000, C=params.get("C", 1.0))
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Training
            model.fit(X_train_scaled, y_train)

            # Cross-Validation
            cv_scores = []
            if do_cv:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5).tolist()

            # Evaluation
            y_pred = model.predict(X_test_scaled)
            
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                "cv_mean": float(np.mean(cv_scores)) if cv_scores else 0.0,
                "cv_std": float(np.std(cv_scores)) if cv_scores else 0.0
            }

            # Optional AUC
            try:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test_scaled)
                    if len(np.unique(y_test)) == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
                    else:
                        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob, multi_class='ovr'))
            except Exception:
                pass

            # Persistence
            model_id = f"mod_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            save_dir = os.path.join(self.storage_path, model_id)
            os.makedirs(save_dir, exist_ok=True)
            
            model_path = os.path.join(save_dir, "model.pkl")
            scaler_path = os.path.join(save_dir, "scaler.pkl")
            
            # Wrap scaler and model for easy deployment
            pipeline = {
                "model": model,
                "scaler": scaler,
                "features": features,
                "encoders": self.encoders,
                "target_column": target_column,
                "model_type": model_type
            }
            
            joblib.dump(pipeline, model_path)

            logger.info("Model training complete", model_id=model_id, accuracy=metrics["accuracy"])

            return {
                "model_id": model_id,
                "status": "success",
                "metrics": metrics,
                "hyperparams": params,
                "features": features,
                "target": target_column,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
                "model_path": model_path,
                "corpus_metadata": {
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "features_count": len(features)
                }
            }

        except Exception as e:
            logger.error("Training failed", error=str(e))
            return {
                "status": "error",
                "message": str(e)
            }
