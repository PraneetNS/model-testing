import pytest
import logging
import pandas as pd
from unittest.mock import patch, MagicMock

from ml_guard.plugins.data_connectors.s3 import S3Connector
from ml_guard.plugins.data_connectors.snowflake import SnowflakeConnector

def test_s3_connector_masks_credentials(caplog):
    connector = S3Connector()
    config = {
        "bucket": "my-bucket",
        "key": "data.csv",
        "region": "us-east-1",
        "aws_access_key_id": "SUPER_SECRET_KEY",
        "aws_secret_access_key": "SUPER_SECRET_TOKEN"
    }

    valid, errors = connector.validate_config(config)
    assert valid

    with patch("ml_guard.plugins.data_connectors.s3.boto3.Session") as mock_session, \
         patch("ml_guard.plugins.data_connectors.s3.pd.read_csv") as mock_read_csv, \
         patch("ml_guard.plugins.data_connectors.s3.tempfile.NamedTemporaryFile") as mock_temp, \
         patch("ml_guard.plugins.data_connectors.s3.os.unlink"):
        
        mock_s3 = MagicMock()
        mock_session.return_value.client.return_value = mock_s3
        
        mock_temp.return_value.__enter__.return_value.name = "/tmp/fake.csv"
        mock_read_csv.return_value = pd.DataFrame({"col1": [1, 2]})

        with caplog.at_level(logging.INFO):
            with patch.object(connector, "save_to_temp", return_value="/tmp/test_saved.csv"):
                # We also need to mock os.path.exists for unlink
                with patch("ml_guard.plugins.data_connectors.s3.os.path.exists", return_value=True):
                    result = connector.fetch(config)
        
        log_output = caplog.text
        assert "SUPER_SECRET_KEY" not in log_output
        assert "SUPER_SECRET_TOKEN" not in log_output
        assert "***" in log_output


def test_snowflake_enforces_row_limit(caplog):
    connector = SnowflakeConnector()
    config = {
        "account": "acc",
        "user": "usr",
        "password": "SUPER_SECRET_PWD",
        "warehouse": "wh",
        "database": "db",
        "schema": "sch",
        "query": "SELECT * FROM giant_table"
    }

    valid, errors = connector.validate_config(config)
    assert valid
    
    with patch("ml_guard.plugins.data_connectors.snowflake.snowflake.connector.connect") as mock_connect, \
         patch("ml_guard.plugins.data_connectors.snowflake.pd.read_sql") as mock_read_sql:
        
        # Simulate returning 100,001 rows
        large_df = pd.DataFrame({"col": range(100001)})
        mock_read_sql.return_value = large_df
        
        with caplog.at_level(logging.INFO):
            with patch.object(connector, "save_to_temp") as mock_save:
                mock_save.return_value = "/tmp/sf_saved.csv"
                connector.fetch(config)
                
                # Check limiting
                saved_df = mock_save.call_args[0][0]
                assert len(saved_df) == 100_000
                
                # Verify the query has LIMIT 100000 appended
                executed_query = mock_read_sql.call_args[0][0]
                assert "LIMIT 100000" in executed_query
        
        log_output = caplog.text
        assert "SUPER_SECRET_PWD" not in log_output
        assert "***" in log_output
