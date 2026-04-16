import logging
import pandas as pd
from typing import Dict, Any, Tuple, List
from .base import DataConnector
import time
from filelock import FileLock
import os
import tempfile

logger = logging.getLogger(__name__)

class OpenMLConnector(DataConnector):
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        if "dataset_id" not in config:
            return False, ["Missing required field: dataset_id"]
        return True, []

    def fetch(self, config: Dict[str, Any]) -> str:
        """
        Fetches a dataset and returns the CSV path. 
        Note: The user asked to return "DataFrame + metadata" in the task description.
        Since fetch() must return a str (the CSV path), we can save the DF and optionally
        the metadata logic can be handled separately, or we just save the dataframe.
        """
        import openml
        
        masked = self.mask_config(config)
        logger.info(f"OpenML fetch initiated: {masked}")

        dataset_id = int(config["dataset_id"])

        # Enforce rate limits: 1-second delay between requests
        lock_path = "/tmp/openml_api.lock"
        if os.name == 'nt':
            lock_path = os.path.join(tempfile.gettempdir(), "openml_api.lock")

        with FileLock(lock_path, timeout=60):
            time.sleep(1) # Ensure 1-second delay
            dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=True, download_features_meta_data=True)

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        
        # Combine X and y into one dataframe
        if y is not None:
            df = pd.concat([X, y], axis=1)
        else:
            df = X

        if len(df) > 100_000:
            logger.warning(f"OpenML dataset has {len(df)} rows, limiting to 100k.")
            df = df.head(100_000)

        # We can extract metadata, though the standard fetch returns just the path
        metadata = {
            "name": dataset.name,
            "description": dataset.description,
            "number_of_classes": dataset.qualities.get("NumberOfClasses", 0) if dataset.qualities else 0,
            "number_of_features": dataset.qualities.get("NumberOfFeatures", 0) if dataset.qualities else 0,
            "number_of_instances": dataset.qualities.get("NumberOfInstances", 0) if dataset.qualities else 0,
            "qualities": dataset.qualities
        }
        logger.info(f"OpenML metadata: {metadata}")

        return self.save_to_temp(df)
