import asyncio
import sys
import os
from sqlalchemy.future import select

# Add the current directory to sys.path
sys.path.insert(0, os.getcwd())

from app.db.session import SessionLocal
from app.db.models import Model, Experiment, GuardrailConfigModel

async def check():
    async with SessionLocal() as db:
        res = await db.execute(select(Model))
        models = res.scalars().all()
        print(f"Models found: {len(models)}")
        for m in models:
            print(f" - ID: {m.id}, Name: {m.name}")
        
        res = await db.execute(select(Experiment))
        experiments = res.scalars().all()
        print(f"\nExperiments found: {len(experiments)}")
        for e in experiments:
            print(f" - ID: {e.id}, Name: {e.name}, ModelID: {e.model_id}, Status: {e.status}")
        
        res = await db.execute(select(GuardrailConfigModel))
        guardrails = res.scalars().all()
        print(f"\nGuardrails found: {len(guardrails)}")
        for g in guardrails:
            print(f" - ID: {g.id}, Name: {g.name}, ModelID: {g.model_id}")

if __name__ == "__main__":
    asyncio.run(check())
