import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add the project root and backend to sys.path
sys.path.append(os.path.abspath("ml_guard/backend"))

from app.db.session import AsyncSessionLocal
from app.db.models import Model, PredictionLog, DriftReport, PerformanceSnapshot, ScanRecord
from sqlalchemy.future import select
from sqlalchemy import func

async def debug_feed():
    async with AsyncSessionLocal() as db:
        # Get all unique model IDs from registry
        try:
            res = await db.execute(select(Model.id))
            model_ids = [str(r) for r in res.scalars().all()]
            print(f"Model IDs from Model table: {model_ids}")
        except Exception as e:
            print(f"Error fetching Model IDs: {e}")
            model_ids = []

        # Supplement with IDs from logs
        try:
            res = await db.execute(select(PredictionLog.model_id).distinct())
            log_ids = [str(r) for r in res.scalars().all()]
            print(f"Model IDs from PredictionLog: {log_ids}")
        except Exception as e:
            print(f"Error fetching log IDs: {e}")
            log_ids = []

        model_ids = list(set(model_ids + [mid for mid in log_ids if mid]))
        print(f"Combined Model IDs: {model_ids}")

        for mid in model_ids:
            print(f"Processing model: {mid}")
            try:
                # Count predictions
                cutoff_24h = datetime.utcnow() - timedelta(hours=24)
                count_stmt = select(func.count(PredictionLog.id)).filter(PredictionLog.model_id == mid, PredictionLog.timestamp >= cutoff_24h)
                pred_count = (await db.execute(count_stmt)).scalar() or 0
                print(f"  Pred count: {pred_count}")

                # Check drift
                drift_stmt = select(DriftReport).filter(DriftReport.model_id == mid).order_by(DriftReport.created_at.desc()).limit(1)
                last_drift = (await db.execute(drift_stmt)).scalars().first()
                print(f"  Last drift: {last_drift}")

                # Check scan records for base score
                last_scan_stmt = select(ScanRecord).filter(ScanRecord.model_id == mid).order_by(ScanRecord.created_at.desc()).limit(1)
                last_scan = (await db.execute(last_scan_stmt)).scalars().first()
                print(f"  Last scan: {last_scan}")

            except Exception as e:
                print(f"  Error processing {mid}: {e}")

if __name__ == "__main__":
    asyncio.run(debug_feed())
