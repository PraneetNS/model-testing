import asyncio
from app.db.session import engine, Base
from app.db.models import RagTrace

async def apply_migrations():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Migration complete!")

if __name__ == "__main__":
    asyncio.run(apply_migrations())
