import importlib
from datetime import datetime, timezone

class WandbSyncPlugin:
    def __init__(self, config: dict):
        self.api_key = config.get("api_key")
        self.entity = config.get("entity")
        self.project = config.get("project")
        self.run_id = config.get("run_id")
        self.metric_map = config.get("metric_map", {})
        
    def pull_run_metrics(self, run_id: str) -> dict:
        try:
            wandb = importlib.import_module("wandb")
            import os
            if self.api_key:
                os.environ["WANDB_API_KEY"] = self.api_key
            
            api = wandb.Api()
            run_path = f"{self.entity}/{self.project}/{run_id}" if self.entity else f"{self.project}/{run_id}"
            run = api.run(run_path)
            
            # Extract scalar metrics from summary
            mapped_metrics = {}
            for k, v in run.summary.items():
                if isinstance(v, (int, float)):
                    target_key = self.metric_map.get(k, k)
                    mapped_metrics[target_key] = v
                
            return mapped_metrics
        except ImportError:
            raise ImportError("wandb is not installed. Install with: pip install wandb")
        except Exception as e:
            raise ValueError(f"W&B connection or auth error: {str(e)}")

    async def sync_to_model(self, model_id: str, run_id: str, db):
        metrics = self.pull_run_metrics(run_id)
        from app.db.models import PerformanceSnapshot
        
        # Upsert: we add a new snapshot for the run
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
