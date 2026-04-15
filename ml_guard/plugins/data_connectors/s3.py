"""
s3.py — Amazon S3 Data Connector for ML Guard
"""
import logging
import os
import tempfile
import boto3
import pandas as pd
from typing import Dict, List, Any, Tuple
from .base import DataConnector

logger = logging.getLogger(__name__)

class S3Connector(DataConnector):
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        required = ["bucket", "key", "region"]
        errors = []
        for field in required:
            if not config.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Credentials must be provided if not using IAM role
        if not (config.get("aws_access_key_id") and config.get("aws_secret_access_key")):
            # We assume it might use ambient credentials, but usually for this 
            # kind of plugin, we want explicit ones or assume_role
            pass
            
        return len(errors) == 0, errors

    def fetch(self, config: Dict[str, Any]) -> str:
        masked = self.mask_config(config)
        logger.info(f"S3 fetch initiated: {masked}")

        bucket = config["bucket"]
        key = config["key"]
        region = config["region"]
        
        session_kwargs = {
            "region_name": region
        }
        
        if config.get("aws_access_key_id"):
            session_kwargs["aws_access_key_id"] = config["aws_access_key_id"]
        if config.get("aws_secret_access_key"):
            session_kwargs["aws_secret_access_key"] = config["aws_secret_access_key"]

        session = boto3.Session(**session_kwargs)
        
        # Handle Assume Role if provided
        if config.get("assume_role_arn"):
            logger.info(f"Assuming role: {config['assume_role_arn']}")
            sts = session.client("sts")
            assumed_role = sts.assume_role(
                RoleArn=config["assume_role_arn"],
                RoleSessionName="MLGuardDataPull"
            )
            creds = assumed_role["Credentials"]
            session = boto3.Session(
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
                region_name=region
            )

        s3 = session.client("s3")
        
        # Create a local copy
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(key)[1]) as tmp:
            s3.download_file(bucket, key, tmp.name)
            local_raw_path = tmp.name

        try:
            # Auto-detect format
            if key.endswith(".parquet"):
                df = pd.read_parquet(local_raw_path)
            else:
                df = pd.read_csv(local_raw_path)
            
            return self.save_to_temp(df)
        finally:
            if os.path.exists(local_raw_path):
                os.unlink(local_raw_path)
