from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import Model, Dataset, NLPIntent, Job

import os
import uuid
import json

router = APIRouter()

@router.post("/initialize")
async def initialize_scan(
    model_name: str = Form(...),
    provider: str = Form("Local"),
    nlp_intent: str = Form(...),
    file_model: UploadFile = File(None),
    file_train: UploadFile = File(None),
    file_test: UploadFile = File(None),
    train_dataset_url: str = Form(None),
    test_dataset_url: str = Form(None),
    db: AsyncSession = Depends(get_db)
):
    if not nlp_intent or nlp_intent.strip() == "":
        raise HTTPException(status_code=400, detail="NLP objective required before governance scan.")

        
    parsed_modules = {"preflight": True, "drift": False, "performance": False, "fairness": False, "llm": False}
    intent_lower = nlp_intent.lower()
    if "drift" in intent_lower: parsed_modules["drift"] = True
    if "perform" in intent_lower or "accuracy" in intent_lower: parsed_modules["performance"] = True
    if "fairness" in intent_lower or "bias" in intent_lower: parsed_modules["fairness"] = True
    if "hallucinat" in intent_lower or "llm" in intent_lower or "toxic" in intent_lower: parsed_modules["llm"] = True
    
    # 1. Create Model
    model = Model(name=model_name, provider=provider)
    db.add(model)
    db.flush()
    
    # 2. Add intent
    intent = NLPIntent(model_id=model.id, raw_intent=nlp_intent, parsed_constraints=parsed_modules)
    db.add(intent)
    
    # ─── 3. Upload Artifacts to Cloud Storage (MinIO) ───
    from app.services.storage_service import upload_dataset as cloud_upload_dataset, upload_model as cloud_upload_model
    
    train_storage_url, test_storage_url = None, None
    train_key, test_key, model_key = None, None, None
    
    if file_model:
        res = cloud_upload_model(file_model.file, file_model.filename, model_id=str(model.id))
        model_key = res["object_key"]
        model.artifact_url = res["url"]
        model.artifact_storage_provider = res["storage_provider"]
        model.artifact_size = res["size"]
        db.flush()
    
    if file_train:
        # Stream directly to cloud
        res = cloud_upload_dataset(file_train.file, file_train.filename, dataset_type="training", scan_id=str(model.id))
        train_storage_url = res["url"]
        train_key = res["object_key"]
        db.add(Dataset(
            model_id=model.id, 
            type="train", 
            metadata_json={"filename": file_train.filename, "key": train_key, "url": train_storage_url},
            row_count=0
        ))
    elif train_dataset_url:
        train_storage_url = train_dataset_url
        train_key = train_dataset_url # Pass as key if it is minio://
        db.add(Dataset(
            model_id=model.id,
            type="train",
            metadata_json={"url": train_dataset_url},
            row_count=0
        ))
        
    if file_test:
        # Stream directly to cloud
        res = cloud_upload_dataset(file_test.file, file_test.filename, dataset_type="testing", scan_id=str(model.id))
        test_storage_url = res["url"]
        test_key = res["object_key"]
        db.add(Dataset(
            model_id=model.id, 
            type="test", 
            metadata_json={"filename": file_test.filename, "key": test_key, "url": test_storage_url},
            row_count=0
        ))
    elif test_dataset_url:
        test_storage_url = test_dataset_url
        test_key = test_dataset_url
        db.add(Dataset(
            model_id=model.id,
            type="test",
            metadata_json={"url": test_dataset_url},
            row_count=0
        ))
    
    db.flush()
    
    # 4. Create Job
    job = Job(model_id=model.id, status="RUNNING")
    db.add(job)
    db.flush()
    
    # Dispatch Celery Task here — passing cloud keys
    from app.workers.tasks import run_comprehensive_scan
    from app.core.celery_app import encrypt_task_payload
    
    payload = {
        "job_id": str(job.id),
        "model_id": str(model.id),
        "modules": parsed_modules,
        "train_path": None,
        "test_path": None,
        "model_artifact_key": model_key,
        "train_dataset_key": train_key,
        "val_dataset_key": test_key
    }
    encrypted_payload = encrypt_task_payload(
        payload, 
        ["train_path", "test_path", "model_artifact_key", "train_dataset_key", "val_dataset_key"]
    )
    
    run_comprehensive_scan.delay(**encrypted_payload)
    
    await db.commit()
    
    return {
        "model_id": str(model.id),
        "job_id": str(job.id),
        "parsed_modules": parsed_modules
    }

