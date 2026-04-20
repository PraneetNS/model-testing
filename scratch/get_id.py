import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "ml_guard", "backend"))

from app.db.session import SessionLocal
from app.db.models import Model
from sqlalchemy import select

async def get_model_id():
    async with SessionLocal() as db:
        result = await db.execute(select(Model))
        model = result.scalars().first()
        if model:
            print(model.id)
        else:
            print("No model found")

if __name__ == "__main__":
    asyncio.run(get_model_id())
