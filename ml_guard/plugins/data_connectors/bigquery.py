"""
bigquery.py — BigQuery Data Connector for ML Guard
"""
import logging
import base64
import json
import pandas as pd
from typing import Dict, List, Any, Tuple
from google.cloud import bigquery
from google.oauth2 import service_account
from .base import DataConnector

logger = logging.getLogger(__name__)

class BigQueryConnector(DataConnector):
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        required = ["project", "query", "service_account_json"]
        errors = []
        for field in required:
            if not config.get(field):
                errors.append(f"Missing required field: {field}")
        return len(errors) == 0, errors

    def fetch(self, config: Dict[str, Any]) -> str:
        masked = self.mask_config(config)
        logger.info(f"BigQuery fetch initiated: {masked}")

        sa_json_b64 = config["service_account_json"]
        sa_info = json.loads(base64.b64decode(sa_json_b64).decode("utf-8"))
        creds = service_account.Credentials.from_service_account_info(sa_info)
        
        client = bigquery.Client(credentials=creds, project=config["project"])
        
        query = config["query"].strip()
        # Enforce 100k row limit
        if "LIMIT" not in query.upper():
            query = f"{query} LIMIT 100000"

        logger.info(f"Executing BQ query: {query}")
        query_job = client.query(query)
        df = query_job.to_dataframe()
        
        if len(df) > 100_000:
            df = df.head(100_000)

        return self.save_to_temp(df)
