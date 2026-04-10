"""
Data Quality Router.
Endpoints for dataset validation before training.
"""
import io
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.core.auth import AuthContext, require_role

router = APIRouter()


@router.post("/data-quality/validate")
async def validate_dataset(
    dataset_file: UploadFile = File(...),
    target_column: str = Form(""),
    reference_file: UploadFile = File(None),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Validate a dataset and return a quality report."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    try:
        from ml_guard.core.data_quality import validate_dataset as _validate
    except ImportError:
        raise HTTPException(500, "Data quality module not available.")

    # Load primary dataset
    dataset_bytes = await dataset_file.read()
    df = pd.read_csv(io.BytesIO(dataset_bytes))

    # Load reference dataset if provided
    ref_df = None
    if reference_file:
        ref_bytes = await reference_file.read()
        ref_df = pd.read_csv(io.BytesIO(ref_bytes))

    # Run validation
    report = _validate(
        df,
        target_column=target_column if target_column else None,
        reference_df=ref_df,
    )

    # ── Transform for Frontend (Enterprise Aesthetic) ───────────────────────────
    checks = report.get("checks", {})
    total = len(checks)
    passed = sum(1 for c in checks.values() if c.get("status") == "passed")
    critical = sum(1 for c in checks.values() if c.get("status") == "failed")
    
    status = "EXCELLENT" if report["quality_score"] >= 90 else "GOOD" if report["quality_score"] >= 75 else "WARNING" if report["quality_score"] >= 50 else "CRITICAL"

    frontend_report = {}
    for check_name, check_data in checks.items():
        s = check_data.get("status", "skipped")
        frontend_report[check_name] = {
            "status": "PASS" if s == "passed" else "FAIL",
            "message": f"{check_name.replace('_', ' ').title()} validation {s}.",
            "score": 1.0 if s == "passed" else 0.5 if s == "warning" else 0.0
        }

    return {
        "quality_score": report["quality_score"], # 0-100
        "checks_passed": passed,
        "total_checks": total,
        "critical_issues": critical,
        "status": status,
        "report": frontend_report,
        "details": {
            "row_count": report["row_count"],
            "feature_count": report["feature_count"],
            "schema_hash": report["schema_hash"]
        }
    }
