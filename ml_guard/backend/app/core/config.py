from typing import List, Union, Optional, Dict, Any
from pydantic import AnyHttpUrl, validator
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """
    Application Settings — reads from environment variables / .env file.
    All secrets MUST come from environment; never hardcode in source.
    """
    PROJECT_NAME: str = "ML Guard Governance Platform"
    API_V1_STR: str = "/api/v1"

    # ── Security ──────────────────────────────────────────────────────────────
    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION_use_openssl_rand_hex_32"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    ALGORITHM: str = "HS256"

    # ── Google OAuth ────────────
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GOOGLE_REDIRECT_URL: str = "http://localhost:8001/api/v1/auth/google/callback"

    # ── Firebase Auth ────────────
    FIREBASE_PROJECT_ID: Optional[str] = "ml-guard"
    FIREBASE_CREDENTIALS_JSON: Optional[str] = None # Path to service account JSON or raw JSON string

    # ── CORS (comma-separated in env: "http://a.com,http://b.com") ────────────
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
    ]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        return v

    # ── Database ──────────────────────────────────────────────────────────────
    # Priority: DATABASE_URL env var > SQLALCHEMY_DATABASE_URI > individual Postgres vars > SQLite fallback
    DATABASE_URL: Optional[str] = None  # Neon / Supabase / any PostgreSQL DSN
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "ml_guard"
    POSTGRES_PORT: str = "5432"
    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    @validator("SQLALCHEMY_DATABASE_URI", pre=True, always=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        # 1. Highest priority: DATABASE_URL environment variable (Neon, Supabase, etc.)
        database_url = values.get("DATABASE_URL")
        if database_url:
            # Neon uses "postgres://" which SQLAlchemy needs as "postgresql://"
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)
            return database_url

        # 2. If SQLALCHEMY_DATABASE_URI is explicitly set
        if isinstance(v, str) and v:
            return v

        # 3. Build from individual PostgreSQL components
        user = values.get("POSTGRES_USER")
        password = values.get("POSTGRES_PASSWORD")
        server = values.get("POSTGRES_SERVER")
        port = values.get("POSTGRES_PORT", "5432")
        db = values.get("POSTGRES_DB")
        if all([user, password, server, db]) and server != "localhost":
            return f"postgresql://{user}:{password}@{server}:{port}/{db}"

        # 4. Development fallback: SQLite
        return f"sqlite:///{os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ml_guard.db'))}"

    # ── MinIO Object Storage ─────────────────────────────────────────────────
    MINIO_ROOT_USER: Optional[str] = None
    MINIO_ROOT_PASSWORD: Optional[str] = None
    MINIO_ENDPOINT: str = "http://localhost:9000"
    MINIO_ACCESS_KEY: Optional[str] = None
    MINIO_SECRET_KEY: Optional[str] = None

    @validator("MINIO_ACCESS_KEY", pre=True, always=True)
    def assemble_minio_access(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        return v or values.get("MINIO_ROOT_USER") or "minioadmin"

    @validator("MINIO_SECRET_KEY", pre=True, always=True)
    def assemble_minio_secret(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        return v or values.get("MINIO_ROOT_PASSWORD") or "minioadmin"

    MINIO_BUCKET: str = "mlguard-artifacts"
    MINIO_REGION: str = "us-east-1"
    STORAGE_MAX_UPLOAD_SIZE: int = 2 * 1024 * 1024 * 1024  # 2 GB default

    # ── Redis / Celery ────────────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_URL: Optional[str] = None

    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None

    @validator("CELERY_BROKER_URL", pre=True)
    def assemble_celery_broker(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        if isinstance(v, str) and v:
            return v
        if values.get("REDIS_URL"):
            return values.get("REDIS_URL")
        return f"redis://{values.get('REDIS_HOST', 'localhost')}:{values.get('REDIS_PORT', 6379)}/0"

    @validator("CELERY_RESULT_BACKEND", pre=True)
    def assemble_celery_result(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        if isinstance(v, str) and v:
            return v
        if values.get("REDIS_URL"):
            return values.get("REDIS_URL")
        return f"redis://{values.get('REDIS_HOST', 'localhost')}:{values.get('REDIS_PORT', 6379)}/0"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


settings = Settings()
