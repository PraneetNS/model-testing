import asyncio
import sys
import os
import uuid
from datetime import datetime, timezone
from sqlalchemy.future import select

# Add the current directory to sys.path
sys.path.insert(0, os.getcwd())

from app.db.session import SessionLocal
from app.db.models import Model, Experiment

async def seed_experiments():
    async with SessionLocal() as db:
        res = await db.execute(select(Model).limit(1))
        model = res.scalar_one_or_none()
        if not model:
            print("No models found to link experiments to.")
            return

        print(f"Seeding experiments for model: {model.name} ({model.id})")
        
        experiments = [
            Experiment(
                id=str(uuid.uuid4()),
                model_id=model.id,
                name="XGBoost Baseline Hyperopt",
                status="COMPLETED",
                parameters={"learning_rate": 0.1, "max_depth": 5, "n_estimators": 100},
                metrics={"accuracy": 0.88, "f1": 0.86, "latency_ms": 12.5},
                framework="xgboost",
                started_at=datetime.now(timezone.utc).replace(tzinfo=None),
                completed_at=datetime.now(timezone.utc).replace(tzinfo=None)
            ),
            Experiment(
                id=str(uuid.uuid4()),
                model_id=model.id,
                name="Deep Neural Network Alpha",
                status="RUNNING",
                parameters={"layers": [64, 32, 16], "activation": "relu", "optimizer": "adam"},
                metrics={"current_loss": 0.42},
                framework="tensorflow",
                started_at=datetime.now(timezone.utc).replace(tzinfo=None)
            ),
            Experiment(
                id=str(uuid.uuid4()),
                model_id=model.id,
                name="Random Forest Pruning Test",
                status="FAILED",
                parameters={"n_estimators": 500, "max_features": "sqrt"},
                metrics={},
                framework="sklearn",
                started_at=datetime.now(timezone.utc).replace(tzinfo=None),
                completed_at=datetime.now(timezone.utc).replace(tzinfo=None)
            )
        ]
        
        for e in experiments:
            db.add(e)
        
        await db.commit()
        print("Experiments seeded successfully.")

if __name__ == "__main__":
    asyncio.run(seed_experiments())
