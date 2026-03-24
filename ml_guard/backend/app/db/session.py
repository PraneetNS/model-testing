from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings


def _build_engine():
    uri = settings.SQLALCHEMY_DATABASE_URI

    if uri and uri.startswith("sqlite"):
        # ── SQLite (dev mode) ──
        from sqlalchemy.pool import StaticPool
        engine = create_engine(
            uri,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            pool_pre_ping=True,
        )
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        return engine

    elif uri and "postgresql" in uri:
        # ── PostgreSQL (Neon / Supabase / Standard) ──
        connect_args = {}
        # Enable SSL for cloud databases (Neon, Supabase, etc.)
        if any(host in uri for host in ["neon.tech", "supabase", "amazonaws", "azure"]):
            connect_args["sslmode"] = "require"

        return create_engine(
            uri,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args=connect_args,
        )

    else:
        # ── Generic fallback ──
        return create_engine(
            uri,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_pre_ping=True,
            pool_recycle=1800,
        )


engine = _build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
