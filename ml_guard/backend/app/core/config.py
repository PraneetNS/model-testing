"""
Configuration settings for ML Guard
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Storage
    STORAGE_PATH: str = "./storage"
    MODELS_PATH: str = "./storage/models"
    DATASETS_PATH: str = "./storage/datasets"
    RESULTS_PATH: str = "./storage/results"

    # ML Engine
    MAX_MODEL_SIZE_MB: int = 500
    MAX_DATASET_SIZE_MB: int = 1000
    SUPPORTED_MODEL_FORMATS: List[str] = [".pkl", ".joblib", ".json", ".pt", ".h5"]
    SUPPORTED_DATA_FORMATS: List[str] = [".csv", ".parquet", ".json"]

    # Test Execution
    MAX_PARALLEL_TESTS: int = 4
    TEST_TIMEOUT_SECONDS: int = 300
    DEFAULT_TEST_SUITE: str = "production-readiness"

    # Quality Gate
    QUALITY_GATE_ENABLED: bool = True
    DEFAULT_QUALITY_THRESHOLDS: dict = {
        "accuracy": 0.85,
        "precision": 0.80,
        "recall": 0.75,
        "f1": 0.80,
        "roc_auc": 0.85,
        "missing_values_threshold": 0.05,
        "drift_psi_threshold": 0.10,
        "bias_disparate_impact_threshold": 1.20
    }

    class Config:
        env_file = ".env"
        case_sensitive = False

# Create global settings instance
settings = Settings()

# Ensure storage directories exist
os.makedirs(settings.STORAGE_PATH, exist_ok=True)
os.makedirs(settings.MODELS_PATH, exist_ok=True)
os.makedirs(settings.DATASETS_PATH, exist_ok=True)
os.makedirs(settings.RESULTS_PATH, exist_ok=True)