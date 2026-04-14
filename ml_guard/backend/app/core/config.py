from typing import List, Optional, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator

class Settings(BaseSettings):
    APP_VERSION: str = "7.2.0"
    DEBUG: bool = False
    PROJECT_NAME: str = "ML Guard"
    
    # Database — PostgreSQL for prod, SQLite only allowed in development
    MLGUARD_ENV: str = "production"
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/mlguard"
    
    # Security
    SECRET_KEY: str = "change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Storage
    MINIO_ENDPOINT: str = "http://localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "mlguard"
    MINIO_USE_SSL: bool = False
    MINIO_REGION: str = "us-east-1"
    
    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
    ]
    
    # Anthropic (optional - for LLM summaries)
    ANTHROPIC_API_KEY: str = ""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    @model_validator(mode="after")
    def validate_sqlite_not_in_prod(self) -> 'Settings':
        if "sqlite" in self.DATABASE_URL.lower():
            if self.MLGUARD_ENV.lower() != "development":
                raise RuntimeError("SQLite is not supported in production. Set DATABASE_URL to a PostgreSQL connection string.")
        return self

settings = Settings()
