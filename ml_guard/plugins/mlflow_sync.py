import importlib
from datetime import datetime, timezone

class MLflowSyncPlugin:
    def __init__(self, config: dict):
        self.tracking_uri = config.get("tracking_uri")
        self.experiment_name = config.get("experiment_name")
        self.run_id = config.get("run_id")
        self.metric_map = config.get("metric_map", {})
        
    def pull_run_metrics(self, run_id: str) -> dict:
        try:
            mlflow = importlib.import_module("mlflow")
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            run = client.get_run(run_id)
            metrics = run.data.metrics
            
            mapped_metrics = {}
            for k, v in metrics.items():
                target_key = self.metric_map.get(k, k)
                mapped_metrics[target_key] = v
                
            return mapped_metrics
        except ImportError:
            raise ImportError("MLflow is not installed. Install with: pip install mlflow")
        except Exception as e:
            raise ValueError(f"MLflow connection or auth error: {str(e)}")

    async def sync_to_model(self, model_id: str, run_id: str, db):
        metrics = self.pull_run_metrics(run_id)
        from app.db.models import PerformanceSnapshot
        
        # Upsert: we just add a new snapshot for the run
        snapshot = PerformanceSnapshot(
            model_id=model_id,
            computed_at=datetime.now(timezone.utc),
            task_type="classification", 
            metrics=metrics,
            sample_count=0
        )
        db.add(snapshot)
        await db.commit()
        return {"synced_metrics": len(metrics), "model_id": model_id, "run_id": run_id}
