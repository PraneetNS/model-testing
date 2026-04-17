"""
Patch audit.py: replace the broken Celery-dependent audit/run with an inline version.
"""
import os

audit_path = r"C:\Users\savan\OneDrive\Desktop\real_Fireflink_ML\ml_guard\backend\app\routers\audit.py"

with open(audit_path, "rb") as f:
    content = f.read().decode("utf-8")

START = "# ENDPOINT 3: Full Audit"
END = "# ENDPOINT 4: Policy config preview"

start_idx = content.find(START)
end_idx = content.find(END)

assert start_idx != -1, "Could not find ENDPOINT 3 marker"
assert end_idx != -1, "Could not find ENDPOINT 4 marker"

# Include the surrounding comment block (════ line before it)
# Go back a line to include the === line
before = content[:start_idx]
# trim trailing whitespace / === line from before block  
after = content[end_idx:]

NEW_ENDPOINT_3 = '''# ENDPOINT 3: Full Audit
# ============================================
@router.post("/audit/run")
async def run_audit(
    model_name: str = Form("CreditRiskDetector"),
    label_col: str = Form("target"),
    model_file: UploadFile = File(...),
    train_file: UploadFile = File(None),
    val_file: UploadFile = File(None),
    train_dataset_url: str = Form(None),
    val_dataset_url: str = Form(None),
    selected: list = Form(["drift", "performance", "fairness", "security"]),
    policy_override: str = Form(None),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer"))
):
    from app.services.storage_service import download_from_url
    from app.db.models import Model, Project
    import numpy as np, base64, tempfile, joblib, os as _os

    # --- Resolve or create Model record ---
    model = (await db.execute(select(Model).filter(Model.name == model_name))).scalars().first()
    if not model:
        project = (await db.execute(select(Project).filter(Project.name == "CI/CD Audits"))).scalars().first()
        if not project:
            project = Project(name="CI/CD Audits", org_id=auth.org_id)
            db.add(project)
            await db.flush()
        model = Model(name=model_name, project_id=project.id, created_by=auth.user_id)
        db.add(model)
        await db.flush()

    import uuid
    submission_token = str(uuid.uuid4())
    job = Job(model_id=model.id, status="PENDING", submission_token=submission_token)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    job_id = str(job.id)

    # --- Read uploaded file bytes eagerly ---
    m_bytes = await model_file.read()

    if val_file and val_file.filename:
        v_bytes = await val_file.read()
    elif val_dataset_url:
        v_bytes = download_from_url(val_dataset_url)
    else:
        raise HTTPException(400, "Provide a validation file or validation_dataset_url.")

    if train_file and train_file.filename:
        t_bytes = await train_file.read()
    elif train_dataset_url:
        t_bytes = download_from_url(train_dataset_url)
    else:
        t_bytes = v_bytes  # use val as surrogate train

    # --- Try Celery; fall back to inline if unavailable ---
    celery_ok = False
    try:
        from app.workers.tasks import run_governance_audit_task
        from app.core.celery_app import encrypt_task_payload
        
        payload = {
            "job_id": job_id, "model_id": str(model.id), "checks": selected,
            "model_b64": base64.b64encode(m_bytes).decode(),
            "train_b64": base64.b64encode(t_bytes).decode(),
            "val_b64": base64.b64encode(v_bytes).decode(),
            "model_filename": model_file.filename,
            "train_filename": (train_file.filename if (train_file and train_file.filename) else (train_dataset_url or "train.csv")),
            "val_filename": (val_file.filename if (val_file and val_file.filename) else (val_dataset_url or "val.csv")),
            "label_col": label_col,
            "user_id": auth.user_id if hasattr(auth, "user_id") else None,
            "org_id": auth.org_id if hasattr(auth, "org_id") else None,
            "policy_override": policy_override,
        }
        
        encrypted_payload = encrypt_task_payload(
            payload,
            ["model_path", "train_path", "val_path"]
        )
        
        run_governance_audit_task.delay(**encrypted_payload)
        celery_ok = True
    except Exception:
        celery_ok = False

    if celery_ok:
        return {
            "status": "pending", "job_id": job_id,
            "submission_token": submission_token,
            "poll_url": f"/api/v1/gate/result/{submission_token}",
            "message": "Governance audit dispatched to worker.",
        }

    # --- Inline fallback: run audit synchronously within request ---
    tmp_files = []
    try:
        def _write_tmp(data: bytes, suffix: str) -> str:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(data); tmp.close(); tmp_files.append(tmp.name)
            return tmp.name

        msuffix = ".onnx" if model_file.filename.lower().endswith(".onnx") else ".pkl"
        model_path = _write_tmp(m_bytes, msuffix)
        train_path = _write_tmp(t_bytes, ".csv")
        val_path = _write_tmp(v_bytes, ".csv")

        if msuffix == ".onnx":
            from ml_guard.core import ONNXModelWrapper
            model_obj = ONNXModelWrapper(model_path)
        else:
            model_obj = joblib.load(model_path)
        if isinstance(model_obj, dict):
            model_obj = next(iter(model_obj.values()))

        def _read_df(path):
            try:
                return pd.read_csv(path)
            except UnicodeDecodeError:
                return pd.read_csv(path, encoding="latin-1")

        df_train = _read_df(train_path)
        df_val = _read_df(val_path)

        feature_names = [c for c in df_train.columns if c != label_col]
        X_train_df = pd.get_dummies(df_train[feature_names])
        X_val_df = pd.get_dummies(df_val[feature_names]).reindex(columns=X_train_df.columns, fill_value=0)

        if getattr(model_obj, "feature_names_in_", None) is not None:
            expected = list(model_obj.feature_names_in_)
            for f in expected:
                X_train_df.setdefault(f, 0); X_val_df.setdefault(f, 0)
            X_train = X_train_df[expected]; X_val = X_val_df[expected]
        else:
            X_train = X_train_df
            X_val = X_val_df.reindex(columns=X_train.columns, fill_value=0)

        y_train_raw = df_train[label_col].values if label_col in df_train.columns else np.zeros(len(df_train))
        y_val_raw = df_val[label_col].values if label_col in df_val.columns else np.zeros(len(df_val))

        from sklearn.preprocessing import LabelEncoder
        try:
            y_train = y_train_raw.astype(float); y_val = y_val_raw.astype(float)
        except (ValueError, TypeError):
            le = LabelEncoder()
            le.fit(np.concatenate([y_train_raw, y_val_raw]))
            y_train = le.transform(y_train_raw).astype(float)
            y_val = le.transform(y_val_raw).astype(float)

        train_preds = model_obj.predict(X_train.values)
        val_preds = model_obj.predict(X_val.values)
        from ml_guard.core import compute_accuracy, compute_f1
        train_acc = float(compute_accuracy(y_train, train_preds))
        val_acc = float(compute_accuracy(y_val, val_preds))
        try:
            val_f1 = float(compute_f1(y_val, val_preds))
        except Exception:
            val_f1 = 0.0
        metrics = {"accuracy": val_acc, "train_accuracy": train_acc, "f1": val_f1}

        from ml_guard.core.drift import compute_feature_drift_report
        drift_report, _ = compute_feature_drift_report(X_train, X_val)

        top_drifted = sorted(
            [{"feature": k, "psi": v.get("PSI", 0),
              "severity": "CRITICAL" if v.get("PSI", 0) > 0.25 else ("WARNING" if v.get("PSI", 0) > 0.15 else "OK")}
             for k, v in drift_report.items()],
            key=lambda x: x["psi"], reverse=True
        )[:10]

        ov_gap = {"accuracy_gap": train_acc - val_acc}

        from ml_guard.core.drift import compute_target_drift
        try:
            target_drift = compute_target_drift(y_train, y_val)
        except Exception:
            target_drift = {}

        try:
            proba = model_obj.predict_proba(X_val.values)[:, 1]
            calibration = compute_calibration(y_val, proba)
        except Exception:
            calibration = {}

        try:
            leakage = detect_leakage(X_train, y_train)
        except Exception:
            leakage = {}

        gov = compute_governance_score(drift_report=drift_report, overfitting_gap=ov_gap)

        from app.domain.services.risk_engine import RiskEngine
        risk_result = RiskEngine().compute(
            drift_report=drift_report, overfitting_gap=ov_gap,
            governance_score=gov["governance_score"],
            security_checks={}, performance_metrics=metrics,
        )

        from app.domain.services.governance_engine import GovernanceEngine
        eval_ctx = {"metrics": metrics, "drift": drift_report, "overfitting_gap": ov_gap,
                    "governance_score": gov["governance_score"]}
        if policy_override:
            import json
            policy_result = evaluate_policy(**eval_ctx, policy=json.loads(policy_override))
        else:
            policy_result = await GovernanceEngine(db).evaluate_active_policy(metrics=eval_ctx, org_id=auth.org_id)

        try:
            advisories = generate_advisories(drift_report=drift_report, overfitting_gap=ov_gap, metrics=metrics)
        except Exception:
            advisories = []

        with open(model_path, "rb") as mf:
            fingerprint = compute_model_fingerprint(mf)
        complexity = compute_model_complexity(model_obj)

        results_json = {
            "checks_run": selected, "metrics": metrics, "drift": drift_report,
            "overfitting_gap": ov_gap, "governance": gov, "policy": policy_result,
            "calibration": calibration, "leakage": leakage, "target_drift": target_drift,
            "advisories": advisories, "risk_score": risk_result.get("risk_score"),
            "risk_level": risk_result.get("risk_level"), "top_drifted_ranked": top_drifted,
            "top5_drifted_features": [f["feature"] for f in top_drifted[:5]],
            "fingerprint": fingerprint, "complexity": complexity,
        }

        scan = ScanRecord(
            model_id=str(model.id), job_id=job_id, scan_type="audit",
            checks_run=selected, results_json=results_json,
            governance_score=gov["governance_score"],
            risk_score=risk_result.get("risk_score"),
            risk_level=risk_result.get("risk_level"),
            gate_status=policy_result.get("gate_status", "UNKNOWN"),
            triggered_by=auth.user_id if hasattr(auth, "user_id") else None,
            trigger_source="inline",
        )
        db.add(scan)

        job_rec = (await db.execute(select(Job).filter(Job.id == job_id))).scalar_one_or_none()
        if job_rec:
            job_rec.status = "COMPLETED"
        await db.commit()
        await db.refresh(scan)

        return {
            "status": "completed", "scan_id": str(scan.id), "job_id": job_id,
            "governance": gov, "risk_score": risk_result.get("risk_score"),
            "risk_level": risk_result.get("risk_level"), "metrics": metrics,
            "drift": drift_report, "top_drifted_ranked": top_drifted,
            "top5_drifted_features": [f["feature"] for f in top_drifted[:5]],
            "overfitting_gap": ov_gap, "target_drift": target_drift,
            "calibration": calibration, "leakage": leakage,
            "policy": policy_result, "advisories": advisories,
            "fingerprint": fingerprint, "complexity": complexity,
        }

    except Exception as exc:
        logger.exception("Inline audit failed")
        job_rec = (await db.execute(select(Job).filter(Job.id == job_id))).scalar_one_or_none()
        if job_rec:
            job_rec.status = "FAILED"
            job_rec.error = str(exc)
            await db.commit()
        raise HTTPException(500, f"Audit failed: {exc}")
    finally:
        for p in tmp_files:
            try:
                if p and _os.path.exists(p):
                    _os.unlink(p)
            except Exception:
                pass


'''

# Also strip the trailing "════" comment block that was before ENDPOINT 3
# (it's already inside `before` - we want to keep the ════ line there)
new_content = before + NEW_ENDPOINT_3 + after

with open(audit_path, "wb") as f:
    f.write(new_content.encode("utf-8"))

print("audit.py patched successfully!")
print(f"New file size: {len(new_content)} chars")
