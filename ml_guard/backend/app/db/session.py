from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from app.core.config import settings

connect_args = {}
engine_kwargs = {}

if "sqlite" in settings.DATABASE_URL:
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite:///") and "aiosqlite" not in db_url:
        settings.DATABASE_URL = db_url.replace("sqlite:///", "sqlite+aiosqlite:///")
    connect_args = {"check_same_thread": False}
else:
    engine_kwargs = {
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
        "pool_pre_ping": True,
    }

engine = create_async_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    **engine_kwargs
)

AsyncSessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

SessionLocal = AsyncSessionLocal  # Legacy alias for backward compatibility during migration

Base = declarative_base()

async def get_db():
    """Returns an async SQLAlchemy session to be injected into FastAPI routes."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
