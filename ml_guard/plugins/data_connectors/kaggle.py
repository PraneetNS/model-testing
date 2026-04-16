import logging
import pandas as pd
from typing import Dict, Any, Tuple, List
from .base import DataConnector
import os
import zipfile
import tempfile
from filelock import FileLock

logger = logging.getLogger(__name__)

class KaggleConnector(DataConnector):
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        required = ["kaggle_username", "kaggle_key", "dataset_slug"]
        errors = []
        for field in required:
            if not config.get(field):
                errors.append(f"Missing required field: {field}")
        return len(errors) == 0, errors

    def fetch(self, config: Dict[str, Any]) -> str:
        masked = self.mask_config(config)
        logger.info(f"Kaggle fetch initiated: {masked}")

        os.environ["KAGGLE_USERNAME"] = config["kaggle_username"]
        os.environ["KAGGLE_KEY"] = config["kaggle_key"]

        # Ensure rate limit: 1 concurrent download per machine.
        # filelock works across processes on the same machine.
        lock_path = "/tmp/kaggle_download.lock"
        if os.name == 'nt':
            lock_path = os.path.join(tempfile.gettempdir(), "kaggle_download.lock")
            
        with FileLock(lock_path, timeout=600):
            # Kaggle API must be imported after env vars are set
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()

            dataset_slug = config["dataset_slug"]
            file_name = config.get("file_name")

            with tempfile.TemporaryDirectory() as tmpdir:
                if file_name:
                    api.dataset_download_file(dataset_slug, file_name=file_name, path=tmpdir)
                else:
                    api.dataset_download_files(dataset_slug, path=tmpdir, unzip=True)
                
                # Find the target CSV
                target_csv = None
                for root, _, files in os.walk(tmpdir):
                    for f in files:
                        if f.endswith(".csv"):
                            if file_name and f != file_name and not file_name.endswith(".csv"):
                                # If file_name was requested, try to match it
                                pass
                            target_csv = os.path.join(root, f)
                            break
                    if target_csv:
                        break

                if not target_csv:
                    raise FileNotFoundError(f"No CSV found in dataset {dataset_slug}")

                df = pd.read_csv(target_csv)
                
                if len(df) > 100_000:
                    logger.warning(f"Kaggle dataset has {len(df)} rows, limiting to 100k.")
                    df = df.head(100_000)

                return self.save_to_temp(df)
