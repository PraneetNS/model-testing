import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import logging
import sys

# Mock modules that might not be installed in the test env
sys.modules['kaggle'] = MagicMock()
sys.modules['kaggle.api'] = MagicMock()
sys.modules['kaggle.api.kaggle_api_extended'] = MagicMock()
sys.modules['openml'] = MagicMock()
sys.modules['openml.datasets'] = MagicMock()
sys.modules['roboflow'] = MagicMock()

from ml_guard.plugins.data_connectors.kaggle import KaggleConnector
from ml_guard.plugins.data_connectors.openml import OpenMLConnector
from ml_guard.plugins.data_connectors.roboflow import RoboflowConnector

def test_kaggle_connector_limit_and_fetch():
    connector = KaggleConnector()
    config = {
        "kaggle_username": "usr",
        "kaggle_key": "key",
        "dataset_slug": "slug/dataset"
    }
    
    with patch("ml_guard.plugins.data_connectors.kaggle.FileLock"), \
         patch("ml_guard.plugins.data_connectors.kaggle.tempfile.TemporaryDirectory") as mock_tmp, \
         patch("ml_guard.plugins.data_connectors.kaggle.os.walk") as mock_walk, \
         patch("ml_guard.plugins.data_connectors.kaggle.pd.read_csv") as mock_read_csv:
        
        mock_tmp.return_value.__enter__.return_value = "/tmp/dir"
        mock_walk.return_value = [("/tmp/dir", [], ["data.csv"])]
        
        # Simulate > 100k
        mock_read_csv.return_value = pd.DataFrame({"col": range(100001)})
        
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_kaggle_api, \
             patch.object(connector, "save_to_temp", return_value="saved.csv") as mock_save:
            
            res = connector.fetch(config)
            
            assert res == "saved.csv"
            saved_df = mock_save.call_args[0][0]
            assert len(saved_df) == 100000  # Enforce row limit
            
def test_openml_connector_fetch_and_metadata(caplog):
    connector = OpenMLConnector()
    config = {"dataset_id": 123}
    
    with patch("ml_guard.plugins.data_connectors.openml.FileLock"), \
         patch("ml_guard.plugins.data_connectors.openml.time.sleep") as mock_sleep:
        
        mock_dataset = MagicMock()
        mock_dataset.name = "Test Dataset"
        mock_dataset.description = "A dataset"
        mock_dataset.qualities = {"NumberOfInstances": 150000, "NumberOfFeatures": 5, "NumberOfClasses": 2}
        mock_dataset.default_target_attribute = "target"
        
        # Simulate returning > 100k
        X = pd.DataFrame({"col1": range(150000)})
        y = pd.Series(range(150000), name="target")
        mock_dataset.get_data.return_value = (X, y, [False], ["col1"])
        
        # Assign the mock directly to the sys module mock
        sys.modules['openml'].datasets.get_dataset.return_value = mock_dataset
        
        with caplog.at_level(logging.INFO):
            with patch.object(connector, "save_to_temp", return_value="saved.csv") as mock_save:
                res = connector.fetch(config)
                
                assert mock_sleep.call_count == 1
                assert res == "saved.csv"
                
                saved_df = mock_save.call_args[0][0]
                assert len(saved_df) == 100000  # Enforce row limit
                
        # Check metadata logged
        log_txt = caplog.text
        assert "OpenML metadata" in log_txt
        assert "150000" in log_txt
        assert "Test Dataset" in log_txt

def test_roboflow_connector_limit_and_retry():
    connector = RoboflowConnector()
    config = {
        "api_key": "dummy",
        "workspace": "ws",
        "project": "proj",
        "version": "1",
        "format": "csv"
    }

    with patch("ml_guard.plugins.data_connectors.roboflow.time.sleep") as mock_sleep, \
         patch("ml_guard.plugins.data_connectors.roboflow.tempfile.TemporaryDirectory") as mock_tmp, \
         patch("ml_guard.plugins.data_connectors.roboflow.os.path.exists", return_value=True), \
         patch("ml_guard.plugins.data_connectors.roboflow.pd.read_csv") as mock_read_csv:
         
        mock_tmp.return_value.__enter__.return_value = "/tmp/dir"
        
        # Simulate > 100k returned
        huge_df = pd.DataFrame({
            "filename": [f"img_{i}.jpg" for i in range(100001)],
            "class": ["dog"] * 100001
        })
        mock_read_csv.return_value = huge_df
        
        with patch("roboflow.Roboflow") as mock_rf, \
             patch.object(connector, "save_to_temp", return_value="saved.csv") as mock_save:
            
            # Sub-mock to raise Exception("429 RateLimit") on first call, succeed on second
            mock_version = MagicMock()
            mock_version.download.side_effect = [Exception("429 RateLimit Exceeded"), MagicMock()]
            
            mock_project = MagicMock()
            mock_project.version.return_value = mock_version
            
            mock_workspace = MagicMock()
            mock_workspace.project.return_value = mock_project
            
            mock_rf.return_value.workspace.return_value = mock_workspace
            
            res = connector.fetch(config)
            
            assert mock_sleep.call_count == 1 # Slept once due to 429
            assert res == "saved.csv"
            
            saved_df = mock_save.call_args[0][0]
            assert len(saved_df) == 100000
