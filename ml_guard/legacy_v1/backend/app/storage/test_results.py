"""
Test Results Storage - Persist and retrieve test execution results
"""

import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
import structlog

from ..core.config import settings
from ..models.test_run import TestRun

logger = structlog.get_logger(__name__)

class TestResultsStorage:
    """Storage service for test run results."""

    def __init__(self):
        self.results_dir = os.path.join(settings.RESULTS_PATH, "test_runs")
        os.makedirs(self.results_dir, exist_ok=True)

    async def save_test_run(self, test_run: TestRun) -> str:
        """
        Save a test run to storage.

        Args:
            test_run: TestRun object to save

        Returns:
            str: File path where the test run was saved
        """

        # Create filename with timestamp and run_id
        filename = f"{test_run.run_id}.json"
        filepath = os.path.join(self.results_dir, filename)

        # Convert to dict for JSON serialization
        run_data = test_run.dict()

        # Convert datetime objects to ISO strings
        run_data["start_time"] = test_run.start_time.isoformat()
        if test_run.end_time:
            run_data["end_time"] = test_run.end_time.isoformat()

        # Convert TestResult objects to dicts
        run_data["results"] = [result.dict() for result in test_run.results]

        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(run_data, f, indent=2, default=str)

            logger.info(
                "Test run saved to storage",
                run_id=test_run.run_id,
                filepath=filepath
            )

            return filepath

        except Exception as e:
            logger.error(
                "Failed to save test run",
                run_id=test_run.run_id,
                error=str(e)
            )
            raise

    async def load_test_run(self, run_id: str) -> Optional[TestRun]:
        """
        Load a test run from storage.

        Args:
            run_id: Test run identifier

        Returns:
            Optional[TestRun]: Loaded test run or None if not found
        """

        filename = f"{run_id}.json"
        filepath = os.path.join(self.results_dir, filename)

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r') as f:
                run_data = json.load(f)

            # Convert ISO strings back to datetime objects
            run_data["start_time"] = datetime.fromisoformat(run_data["start_time"])
            if run_data.get("end_time"):
                run_data["end_time"] = datetime.fromisoformat(run_data["end_time"])

            # Reconstruct TestRun object
            test_run = TestRun(**run_data)

            logger.debug(
                "Test run loaded from storage",
                run_id=run_id,
                filepath=filepath
            )

            return test_run

        except Exception as e:
            logger.error(
                "Failed to load test run",
                run_id=run_id,
                error=str(e)
            )
            return None

    async def list_test_runs(
        self,
        project_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[TestRun]:
        """
        List test runs with optional filtering.

        Args:
            project_id: Filter by project ID
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List[TestRun]: List of test runs
        """

        all_files = os.listdir(self.results_dir)
        json_files = [f for f in all_files if f.endswith('.json')]

        test_runs = []
        for filename in json_files[offset:offset + limit]:
            run_id = filename.replace('.json', '')
            test_run = await self.load_test_run(run_id)

            if test_run:
                if project_id is None or test_run.project_id == project_id:
                    test_runs.append(test_run)

        # Sort by start time (most recent first)
        test_runs.sort(key=lambda x: x.start_time, reverse=True)

        return test_runs

    async def get_test_run_summary(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a test run without loading full results.

        Args:
            run_id: Test run identifier

        Returns:
            Optional[Dict[str, Any]]: Test run summary
        """

        test_run = await self.load_test_run(run_id)
        if not test_run:
            return None

        return {
            "run_id": test_run.run_id,
            "project_id": test_run.project_id,
            "model_version": test_run.model_version,
            "test_suite": test_run.test_suite,
            "environment": test_run.environment,
            "status": test_run.status,
            "start_time": test_run.start_time.isoformat(),
            "execution_time_seconds": test_run.execution_time_seconds,
            "total_tests": len(test_run.results),
            "passed_tests": test_run.passed_tests,
            "failed_tests": test_run.failed_tests,
            "warning_tests": test_run.warning_tests,
            "deployment_blocked": test_run.deployment_blocked
        }

    async def delete_test_run(self, run_id: str) -> bool:
        """
        Delete a test run from storage.

        Args:
            run_id: Test run identifier

        Returns:
            bool: True if deleted, False if not found
        """

        filename = f"{run_id}.json"
        filepath = os.path.join(self.results_dir, filename)

        if not os.path.exists(filepath):
            return False

        try:
            os.remove(filepath)
            logger.info(
                "Test run deleted from storage",
                run_id=run_id,
                filepath=filepath
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to delete test run",
                run_id=run_id,
                error=str(e)
            )
            return False