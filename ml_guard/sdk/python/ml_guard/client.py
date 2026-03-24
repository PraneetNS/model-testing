import os
import io
import time
import joblib
import pandas as pd
import requests
from typing import Any, Optional, Dict
from .logger import setup_logger
from .detector import ModelDetector

class Guard:
    """
    ML Guard Python SDK for Enterprise Quality Governance.
    """
    def __init__(self, project: str, api_url: Optional[str] = None, api_token: Optional[str] = None):
        self.project = project
        self.api_url = api_url or os.getenv("MLGUARD_URL", "http://localhost:8001/api/v1")
        self.api_token = api_token or os.getenv("MLGUARD_TOKEN")
        self.logger = setup_logger()
        self.detector = ModelDetector()

    def evaluate(
        self, 
        model: Any, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        target_column: str = "target",
        query: Optional[str] = None,
        timeout: int = 300,
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluates a model by uploading artifacts to the ML Guard backend.
        """
        self.logger.info(f"Starting evaluation for project: {self.project}")
        
        model_type = self.detector.detect_type(model)
        self.logger.info(f"Detected model type: {model_type}")

        # Serialize artifacts in memory
        model_buffer = io.BytesIO()
        joblib.dump(model, model_buffer)
        model_buffer.seek(0)

        train_buffer = io.BytesIO()
        train_df.to_csv(train_buffer, index=False)
        train_buffer.seek(0)

        val_buffer = io.BytesIO()
        val_df.to_csv(val_buffer, index=False)
        val_buffer.seek(0)

        files = {
            "model_file": ("model.pkl", model_buffer, "application/octet-stream"),
            "train_file": ("train.csv", train_buffer, "text/csv"),
            "val_file": ("val.csv", val_buffer, "text/csv")
        }

        data = {
            "project_id": self.project,
            "target_column": target_column,
            "query": query or f"Automated CI scan for {model_type}"
        }

        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        last_error = None
        for attempt in range(retries):
            try:
                self.logger.info(f"Uploading artifacts to {self.api_url}/quality-gate/evaluate (Attempt {attempt + 1})")
                response = requests.post(
                    f"{self.api_url}/quality-gate/evaluate",
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.logger.info("Evaluation successful", extra={
                        "score": result.get("score"),
                        "deployment_allowed": result.get("deployment_allowed"),
                        "risk_level": result.get("risk_level")
                    })
                    
                    if not result.get("deployment_allowed"):
                        self.logger.error("QUALITY GATE FAILED: Deployment not allowed based on risk score.")
                        # we don't exit here, we return the result so the user can handle it or the CLI wrapper can exit.
                    
                    return result
                
                else:
                    self.logger.warning(f"Backend returned error: {response.status_code} - {response.text}")
                    last_error = f"HTTP {response.status_code}: {response.text}"
            
            except Exception as e:
                self.logger.error(f"Request failed: {str(e)}")
                last_error = str(e)
            
            if attempt < retries - 1:
                time.sleep(2 ** attempt) # Exponential backoff

        raise RuntimeError(f"Failed to evaluate model after {retries} attempts. Last error: {last_error}")
