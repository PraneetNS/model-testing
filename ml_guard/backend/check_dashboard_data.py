import asyncio
from app.db.session import SessionLocal
from app.db.models import PredictionLog, Model
from sqlalchemy import func, select

async def check_data():
    db = SessionLocal()
    try:
        # Check models
        m_res = await db.execute(select(func.count(Model.id)))
        m_count = m_res.scalar()
        
        # Check predictions
        p_res = await db.execute(select(func.count(PredictionLog.id)))
        p_count = p_res.scalar()
        
        print(f"Registered Models: {m_count}")
        print(f"Total Prediction Logs: {p_count}")
        
        if p_count > 0:
            # Check latest prediction timestamp
            latest_res = await db.execute(select(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(1))
            latest = latest_res.scalars().first()
            print(f"Latest Prediction at: {latest.timestamp}")
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(check_data())
