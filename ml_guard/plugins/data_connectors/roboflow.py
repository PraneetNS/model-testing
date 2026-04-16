import logging
import pandas as pd
from typing import Dict, Any, Tuple, List
from .base import DataConnector
import time
import tempfile
import os

logger = logging.getLogger(__name__)

class RoboflowConnector(DataConnector):
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        required = ["api_key", "workspace", "project", "version", "format"]
        errors = []
        for field in required:
            if not config.get(field):
                errors.append(f"Missing required field: {field}")
        
        if config.get("format") not in ["yolov8", "coco", "csv"]:
            errors.append("Invalid format. Must be yolov8, coco, or csv.")
            
        return len(errors) == 0, errors

    def fetch(self, config: Dict[str, Any]) -> str:
        masked = self.mask_config(config)
        logger.info(f"Roboflow fetch initiated: {masked}")

        # The roboflow package rate limits can be handled by catching 429 exceptions 
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=config["api_key"])
        
        # Exponential backoff on rate limit
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # The upstream API limits are implicitly respected via retries if 429
                workspace = rf.workspace(config["workspace"])
                project = workspace.project(config["project"])
                version = project.version(config["version"])
                
                # Fetch dataset download link and process
                # We would normally download locally, extract images and CSV
                with tempfile.TemporaryDirectory() as tmpdir:
                    dataset = version.download(config["format"], location=tmpdir)
                    
                    # For governance drift/fairness on CV, convert annotations to tabular
                    rows = []
                    import os
                    # Simplified annotation parsing logic
                    # Assuming format CSV was chosen or converting from yolov8/coco
                    csv_path = os.path.join(tmpdir, "train", "_annotations.csv")
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        # Ensure standard cols: image_id, class_label, confidence, split
                        if "filename" in df.columns and "class" in df.columns:
                            rows = [{"image_id": r["filename"], "class_label": r["class"], "confidence": 1.0, "split": "train"} for _, r in df.iterrows()]
                    else:
                        # Fallback simple iteration across dirs to simulate parsed data
                        for split in ["train", "valid", "test"]:
                            split_dir = os.path.join(tmpdir, split)
                            if os.path.isdir(split_dir):
                                for f in os.listdir(split_dir):
                                    if f.endswith(".jpg") or f.endswith(".png"):
                                        rows.append({
                                            "image_id": f,
                                            "class_label": "unknown", # Mock parsed
                                            "confidence": 1.0,
                                            "split": split
                                        })
                    
                    final_df = pd.DataFrame(rows)
                    if len(final_df) == 0:
                        final_df = pd.DataFrame(columns=["image_id", "class_label", "confidence", "split"])
                        
                    if len(final_df) > 100_000:
                        logger.warning(f"Roboflow dataset has {len(final_df)} rows, limiting to 100k.")
                        final_df = final_df.head(100_000)

                    return self.save_to_temp(final_df)

            except Exception as e:
                if "429" in str(e) or "RateLimit" in str(e):
                    if attempt < max_retries - 1:
                        sleep_time = 2 ** attempt
                        logger.warning(f"Roboflow rate limit hit. Retrying in {sleep_time}s.")
                        time.sleep(sleep_time)
                        continue
                raise
        
        raise RuntimeError("Roboflow fetch failed after retries.")
