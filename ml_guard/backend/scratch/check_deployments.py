
import asyncio
import os
import sys
import uuid

# Add backend to path
sys.path.append(os.getcwd())

from app.db.session import SessionLocal
from app.db.models import Deployment, ModelVersion, Model, Environment, User
from sqlalchemy import select, func

async def check():
    async with SessionLocal() as db:
        # Check counts
        deploys = (await db.execute(select(Deployment))).scalars().all()
        print(f"Total Deployments in DB: {len(deploys)}")
        for d in deploys:
            print(f"ID: {d.id}, Env: {d.environment}, Status: {d.status}, Version: {d.version_id}")
        
        # Check if environment names match activeTab expectations
        envs = (await db.execute(select(Environment))).scalars().all()
        print(f"Total Environments in DB: {len(envs)}")
        for e in envs:
            print(f"Env Name: {e.name}, Active: {e.is_active}")

if __name__ == "__main__":
    asyncio.run(check())
