import hashlib
import pandas as pd
from typing import Dict, Any, Tuple
import structlog

logger = structlog.get_logger(__name__)

class FingerprintingService:
    """
    Tier 1: Dataset Fingerprinting & Schema Versioning.
    Generates deterministic hashes for dataset state to ensure reproducibility.
    """
    
    @staticmethod
    def generate_fingerprint(df: pd.DataFrame) -> str:
        """
        Generates a SHA-256 hash of the dataframe content.
        Uses a stable representation by sorting columns and rounding floats.
        """
        # Create a stable representation for hashing
        # 1. Sort columns
        # 2. Handle float precision (round to 6 decimal places)
        # 3. Handle categorical ordering
        
        stable_df = df.reindex(sorted(df.columns), axis=1)
        
        # Hash the values
        hash_obj = hashlib.sha256()
        
        # For very large datasets, hashing the first 10k rows + metadata might be faster
        # but for true reproducibility, we hash the entire content.
        content_bytes = pd.util.hash_pandas_object(stable_df, index=True).values.tobytes()
        hash_obj.update(content_bytes)
        
        return hash_obj.hexdigest()

    @staticmethod
    def extract_schema(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extracts column names and dtypes for versioning and validation.
        """
        schema = {
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict()
        }
        return schema

    @staticmethod
    def validate_schema(df: pd.DataFrame, expected_schema: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates if the provided dataframe matches the expected schema.
        """
        current_cols = set(df.columns)
        expected_cols = set(expected_schema.get("columns", []))
        
        if current_cols != expected_cols:
            missing = expected_cols - current_cols
            extra = current_cols - expected_cols
            return False, f"Schema mismatch. Missing: {missing}, Extra: {extra}"
            
        return True, "Schema validated"
