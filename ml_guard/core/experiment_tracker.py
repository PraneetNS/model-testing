
import time
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ExperimentRun:
    """In-memory experiment run tracker."""

    def __init__(self, experiment_name, model_name=None, parameters=None):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.parameters = parameters or {}
        self.metrics = {}
        self.artifacts = []
        self.tags = {}
        self.status = "RUNNING"
        self.start_time = time.time()
        self.end_time = None

    def log_param(self, key, value):
        self.parameters[key] = value

    def log_params(self, params: dict):
        self.parameters.update(params)

    def log_metric(self, key, value):
        self.metrics[key] = float(value)

    def log_metrics(self, metrics: dict):
        for k, v in metrics.items():
            self.metrics[k] = float(v)

    def log_artifact(self, artifact_url):
        self.artifacts.append(artifact_url)

    def set_tag(self, key, value):
        self.tags[key] = value

    def end(self, status="COMPLETED"):
        self.status = status
        self.end_time = time.time()

    def to_dict(self):
        elapsed_ms = int((self.end_time or time.time()) - self.start_time) * 1000
        return {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "tags": self.tags,
            "status": self.status,
            "training_time_ms": elapsed_ms,
            "started_at": datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat(),
            "completed_at": datetime.fromtimestamp(self.end_time, tz=timezone.utc).isoformat() if self.end_time else None,
        }


def extract_model_params(model):
    """Auto-extract hyperparameters from a scikit-learn model."""
    params = {}
    try:
        if hasattr(model, "get_params"):
            raw = model.get_params(deep=False)
            for k, v in raw.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    params[k] = v
                else:
                    params[k] = str(v)
    except Exception as e:
        logger.warning("Could not extract model parameters: %s", e)
    return params


def extract_training_metrics(model, X_train, y_train, X_test=None, y_test=None):
    """Compute standard training metrics from a fitted model."""
    metrics = {}
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        if hasattr(model, "predict"):
            y_pred_train = model.predict(X_train)
            metrics["train_accuracy"] = float(accuracy_score(y_train, y_pred_train))

            try:
                metrics["train_f1"] = float(f1_score(y_train, y_pred_train, average="weighted", zero_division=0))
                metrics["train_precision"] = float(precision_score(y_train, y_pred_train, average="weighted", zero_division=0))
                metrics["train_recall"] = float(recall_score(y_train, y_pred_train, average="weighted", zero_division=0))
            except Exception:
                pass

            if X_test is not None and y_test is not None:
                y_pred_test = model.predict(X_test)
                metrics["test_accuracy"] = float(accuracy_score(y_test, y_pred_test))
                try:
                    metrics["test_f1"] = float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0))
                except Exception:
                    pass

                # Overfitting gap
                metrics["overfitting_gap"] = round(
                    metrics.get("train_accuracy", 0) - metrics.get("test_accuracy", 0), 4
                )

    except Exception as e:
        logger.warning("Metric extraction failed: %s", e)

    return metrics
