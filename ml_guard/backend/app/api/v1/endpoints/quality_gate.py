
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional
import pandas as pd
import io
import joblib
from app.domain.services.orchestrator import TestOrchestrator
from app.domain.services.nlp_parser import NLPParser
from app.domain.services.trainer import Trainer

router = APIRouter()
orchestrator = TestOrchestrator()
nlp_parser = NLPParser()
trainer_service = Trainer()

@router.post("/evaluate")
async def evaluate_model(
    project_id: str = Form(...),
    model_file: UploadFile = File(...),
    train_file: UploadFile = File(...),
    val_file: UploadFile = File(...),
    target_column: str = Form("churn"),
    query: Optional[str] = Form(None)
):
    """
    Evaluate an uploaded model against uploaded datasets.
    Restores the 'input model and all' functionality.
    """
    try:
        # Load Model
        model_content = await model_file.read()
        model = joblib.load(io.BytesIO(model_content))
        
        # Load Datasets
        train_df = pd.read_csv(io.BytesIO(await train_file.read()))
        val_df = pd.read_csv(io.BytesIO(await val_file.read()))
        
        # Enhanced Dataset Validation
        missing_errors = []
        if target_column not in train_df.columns:
            missing_errors.append(f"Target variable '{target_column}' was not found in the Training dataset. Available columns: {list(train_df.columns[:10])}...")
        if target_column not in val_df.columns:
            missing_errors.append(f"Target variable '{target_column}' was not found in the Validation dataset. Available columns: {list(val_df.columns[:10])}...")
        
        if missing_errors:
            raise HTTPException(status_code=400, detail=" | ".join(missing_errors))

        datasets = {
            "training": train_df,
            "validation": val_df
        }

        # Determine tests from NLP or default
        categories = ["accuracy", "data_quality", "drift"]
        if query:
            categories = nlp_parser.parse_query(query)

        # Run Orchestration
        result = await orchestrator.run_test_suite(
            project_id=project_id,
            model_version="v1.0.0",
            test_suite_name="NLP-Triggered Scan",
            model_artifact=model,
            datasets=datasets,
            categories=categories,
            target_column=target_column
        )

        # Add "AI" Explanations to failed results
        for r in result.results:
            if r.status == "failed":
                r.explanation = _generate_remediation(r)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Engine Error: {str(e)}")

def _generate_remediation(result) -> str:
    """Semi-automated remediation logic based on test type."""
    explanations = {
        "accuracy_threshold": "The model accuracy fell below the safety threshold. Consider checking for label noise or performing hyperparameter tuning to boost generalization.",
        "precision_threshold": "High false positive rate detected. If this is a churn model, you might be targeting loyal customers as 'at risk'. Try adjusting the decision threshold.",
        "missing_values": "Data quality is compromised by null values. Check your data pipeline or upstream SQL transformations for missing join keys.",
        "psi_drift": "Significant distribution shift detected! The environment your model was trained on differs from the current validation set. Retraining with fresh data is required.",
        "class_balance": "Severe class imbalance detected. The minority class is underrepresented, which leads to biased results toward the majority class. Use SMOTE or oversampling techniques.",
        "disparate_impact": "Detecting potential bias in protected groups. Your model may be discriminating based on sensitive attributes like gender or age. Review fairness constraints."
    }
    return explanations.get(result.category if result.category in explanations else result.test_id.split('-')[0], 
                            "Performance degradation detected in this node. Investigating features with high influence is recommended.")

@router.post("/train")
async def train_new_model(
    dataset: UploadFile = File(...),
    target_column: str = Form(...),
    model_type: str = Form("random_forest"),
    test_size: float = Form(0.2),
    do_cv: bool = Form(True)
):
    """
    Enhanced training endpoint with production-grade options.
    """
    try:
        df = pd.read_csv(io.BytesIO(await dataset.read()))
        
        # Validate Dataset
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target variable '{target_column}' not found. Did you forget to include the column or is it a typo?")

        # More realistic training options could be added here
        result = trainer_service.train_model(
            df=df, 
            target_column=target_column, 
            model_type=model_type,
            test_size=test_size,
            do_cv=do_cv
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message", "Training failed"))
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Kernel Error: {str(e)}")
