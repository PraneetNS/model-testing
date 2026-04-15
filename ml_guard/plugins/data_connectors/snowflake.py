"""
snowflake.py — Snowflake Data Connector for ML Guard
"""
import logging
import pandas as pd
import snowflake.connector
from typing import Dict, List, Any, Tuple
from .base import DataConnector

logger = logging.getLogger(__name__)

class SnowflakeConnector(DataConnector):
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        required = ["account", "user", "password", "warehouse", "database", "schema", "query"]
        errors = []
        for field in required:
            if not config.get(field):
                errors.append(f"Missing required field: {field}")
        return len(errors) == 0, errors

    def fetch(self, config: Dict[str, Any]) -> str:
        masked = self.mask_config(config)
        logger.info(f"Snowflake fetch initiated: {masked}")

        conn = snowflake.connector.connect(
            user=config["user"],
            password=config["password"],
            account=config["account"],
            warehouse=config["warehouse"],
            database=config["database"],
            schema=config["schema"]
        )

        try:
            query = config["query"].strip()
            # Enforce 100k row limit
            if "LIMIT" not in query.upper():
                query = f"{query} LIMIT 100000"
            
            logger.info(f"Executing query: {query}")
            df = pd.read_sql(query, conn)
            
            # Additional safety check on row count
            if len(df) > 100_000:
                logger.warning(f"Query returned {len(df)} rows, truncating to 100k.")
                df = df.head(100_000)

            return self.save_to_temp(df)
        finally:
            conn.close()
