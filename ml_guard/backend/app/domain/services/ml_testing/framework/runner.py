from typing import List, Dict, Any
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
from .base import MLTestCaseResult, TestStatus
from .registry import get_test_class

class MLTestReport(BaseModel):
    summary: Dict[str, Any]
    results: List[MLTestCaseResult]
    metadata: Dict[str, Any] = {}

class MLTestRunner:
    """The JUnit-style runner for ML Guard tests."""
    
    async def run_suite(
        self, 
        suite_config: Dict[str, Any], 
        model: Any, 
        datasets: Dict[str, Any], 
        baseline_model: Any = None,
        baseline_datasets: Dict[str, Any] = None
    ) -> MLTestReport:
        
        results = []
        for test_conf in suite_config.get("tests", []):
            test_type = test_conf.get("type")
            test_cls = get_test_class(test_type)
            
            if test_cls:
                test_instance = test_cls(test_conf)
                result = await test_instance.run(
                    model, 
                    datasets, 
                    baseline_model=baseline_model, 
                    baseline_datasets=baseline_datasets
                )
                results.append(result)
            else:
                # Fallback for unregistered tests or errors
                pass

        total = len(results)
        passed = sum(1 for r in results if r.status == TestStatus.PASS)
        failed = sum(1 for r in results if r.status == TestStatus.FAIL)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)
        warnings = sum(1 for r in results if r.status == TestStatus.WARN)
        
        quality_index = (passed / total * 100) if total > 0 else 0
        
        return MLTestReport(
            summary={
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "warnings": warnings,
                "quality_index": round(quality_index, 2)
            },
            results=results,
            metadata={
                "suite_name": suite_config.get("name", "Unnamed Suite"),
                "timestamp": pd.Timestamp.now().isoformat() if 'pd' in globals() else ""
            }
        )
