"""
Seed realistic prediction logs into the database for the Performance module.
"""
import asyncio
import sys, os
sys.path.insert(0, os.getcwd())

import random
from datetime import datetime, timedelta, timezone
from app.db.session import SessionLocal
from app.db.models import PredictionLog
from sqlalchemy.future import select
from sqlalchemy import func

async def seed():
    async with SessionLocal() as db:
        count = (await db.execute(select(func.count()).select_from(PredictionLog))).scalar()
        if count and count > 10:
            print(f"Already have {count} prediction logs — skipping seed.")
            return

        print("Seeding prediction logs...")
        now = datetime.now(timezone.utc)
        predictions_list = ["0", "1", "0", "1", "1", "0"]

        logs = []
        for i in range(60):
            ts = now - timedelta(minutes=random.randint(0, 180))
            confidence = round(random.uniform(0.45, 0.99), 3)
            logs.append(PredictionLog(
                model_id="perf-demo-model",
                latency_ms=round(random.uniform(12.5, 145.0), 2),
                prediction=random.choice(predictions_list),
                confidence=confidence,
                data_source="api",
                environment="production",
                timestamp=ts,
            ))

        db.add_all(logs)
        await db.commit()
        print(f"✅ Seeded {len(logs)} prediction logs.")

if __name__ == "__main__":
    asyncio.run(seed())
