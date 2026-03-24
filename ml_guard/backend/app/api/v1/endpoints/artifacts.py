from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional, Dict, Any
import structlog
from app.api.v1 import deps
from app.domain.services.artifact_inspector import ArtifactInspector
from app.infrastructure.persistence import models as sql_models

logger = structlog.get_logger(__name__)
router = APIRouter()
inspector = ArtifactInspector()

@router.post("/preflight")
async def inspect_artifacts(
    model_file: UploadFile = File(...),
    train_file: UploadFile = File(...),
    val_file: UploadFile = File(...),
    target_column: str = Form("churn"),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """
    Enterprise-grade artifact inspection.
    Performs deep profiling and cross-artifact compatibility checks.
    """
    try:
        model_content = await model_file.read()
        train_content = await train_file.read()
        val_content = await val_file.read()

        # 1. Inspect Artifacts
        model_meta = await inspector.inspect_model(model_content, model_file.filename or "model.pkl")
        train_meta = await inspector.profile_dataset(train_content, train_file.filename or "train.csv", target_column)
        val_meta = await inspector.profile_dataset(val_content, val_file.filename or "val.csv", target_column)

        # 2. Cross-Check Compatibility
        compatibility_errors = inspector.validate_compatibility(model_meta, train_meta)

        return {
            "model": model_meta,
            "train_dataset": train_meta,
            "validation_dataset": {
                "rows": val_meta["rows"],
                "status": val_meta["status"]
            },
            "compatibility": {
                "is_compatible": len(compatibility_errors) == 0,
                "errors": compatibility_errors
            },
            "summary": {
                "compatible": len(compatibility_errors) == 0,
                "message": "Protocol Verification Successful" if len(compatibility_errors) == 0 else "Compatibility Violations Detected"
            }
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail={"success": False, "error": str(ve)})
    except Exception as e:
        logger.error("Deep inspection failure", error=str(e))
        raise HTTPException(status_code=500, detail={"success": False, "error": "Neural telemetry failure"})
