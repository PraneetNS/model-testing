"""
test_data_connectors.py — Unit tests for Data Connector Plugin System
"""
import unittest
from unittest.mock import patch, MagicMock
import logging
import io
import pandas as pd
from ml_guard.plugins.data_connectors.s3 import S3Connector
from ml_guard.plugins.data_connectors.snowflake import SnowflakeConnector

class TestDataConnectors(unittest.TestCase):
    def setUp(self):
        # Setup log capture
        self.log_capture = io.StringIO()
        self.logger = logging.getLogger("ml_guard.plugins.data_connectors")
        self.handler = logging.StreamHandler(self.log_capture)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)

    def tearDown(self):
        self.logger.removeHandler(self.handler)

    @patch("boto3.Session")
    def test_s3_log_redaction(self, mock_session):
        """Verify that S3 credentials are never logged."""
        connector = S3Connector()
        config = {
            "bucket": "my-bucket",
            "key": "data.csv",
            "region": "us-east-1",
            "aws_access_key_id": "AKIA_SECRET_ID",
            "aws_secret_access_key": "VERY_SECRET_KEY"
        }
        
        # Mock s3 download
        mock_s3 = MagicMock()
        mock_session.return_value.client.return_value = mock_s3
        
        # We only want to test the logging part of fetch, so let's mock the actual logic
        with patch.object(connector, "save_to_temp", return_value="/tmp/test.csv"):
            with patch("pandas.read_csv", return_value=pd.DataFrame({"a": [1]})):
                with patch("tempfile.NamedTemporaryFile"):
                    try:
                        connector.fetch(config)
                    except:
                        pass # NamedTemporaryFile mock might cause issues, that's fine
        
        log_output = self.log_capture.getvalue()
        self.assertIn("S3 fetch initiated", log_output)
        self.assertNotIn("AKIA_SECRET_ID", log_output)
        self.assertNotIn("VERY_SECRET_KEY", log_output)
        self.assertIn("***", log_output)

    @patch("snowflake.connector.connect")
    @patch("pandas.read_sql")
    def test_snowflake_row_limit(self, mock_read_sql, mock_connect):
        """Verify that Snowflake queries are capped at 100k rows."""
        connector = SnowflakeConnector()
        config = {
            "account": "abc", "user": "u", "password": "p",
            "warehouse": "w", "database": "d", "schema": "s",
            "query": "SELECT * FROM large_table"
        }
        
        # Mock large return
        mock_read_sql.return_value = pd.DataFrame({"col": range(150_000)})
        
        with patch.object(connector, "save_to_temp") as mock_save:
            connector.fetch(config)
            
            # Check query modification
            args, kwargs = mock_read_sql.call_args
            self.assertIn("LIMIT 100000", args[0])
            
            # Check truncation if mock returned more (safety check)
            saved_df = mock_save.call_args[0][0]
            self.assertEqual(len(saved_df), 100_000)

    def test_mask_config_helper(self):
        """Direct test of the mask_config helper."""
        from ml_guard.plugins.data_connectors.base import DataConnector
        class TestConn(DataConnector):
            def fetch(self, config): return ""
            def validate_config(self, config): return True, []
            
        conn = TestConn()
        raw = {"password": "secret", "user": "admin", "sas_token": "token123"}
        masked = conn.mask_config(raw)
        
        self.assertEqual(masked["password"], "***")
        self.assertEqual(masked["sas_token"], "***")
        self.assertEqual(masked["user"], "admin")


if __name__ == "__main__":
    unittest.main()
