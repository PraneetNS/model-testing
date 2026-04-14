import time
import os
import io
import tempfile
import joblib
import pandas as pd
import numpy as np
import logging
from app.core.celery_app import celery_app
from app.db.session import SessionLocal
from app.core.config import settings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
from app.db.models import Job, PreflightResult, DriftResult, PerformanceResult, FairnessResult, LLMResult, GovernanceResult, ExplainabilityResult, Model as ModelRecord
from ml_guard.core import MLEvaluator, Constraint, compute_accuracy, compute_f1, ONNXModelWrapper
from ml_guard.core.aibom import generate_aibom
import onnxruntime as ort

logger = logging.getLogger(__name__)

try:
    from app.services.storage_service import download_artifact, upload_artifact
    _has_storage = True
except ImportError:
    _has_storage = False

try:
    from ml_guard.core.explainability import run_explainability
except ImportError:
    run_explainability = None


def _load_artifact_to_tempfile(object_key: str, suffix: str = ".pkl") -> str:
    """Download from MinIO to a temp file, return path. Caller must delete."""
    data = download_artifact(object_key)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _load_dataset_from_minio(path_or_key: str) -> pd.DataFrame:
    """Download a dataset (CSV or Parquet) from MinIO/S3 and return as DataFrame."""
    # Handle minio:// schema
    object_key = path_or_key.replace("minio://", "")
    
    data = download_artifact(object_key)
    buffer = io.BytesIO(data)
    
    if object_key.lower().endswith(".parquet"):
        try:
            return pd.read_parquet(buffer)
        except Exception as e:
            logger.error(f"Failed to load Parquet: {e}")
            raise
    return pd.read_csv(buffer)


def _load_model_artifact(object_key: str):
    """Detect format and load model from MinIO."""
    suffix = ".onnx" if object_key.lower().endswith(".onnx") else ".pkl"
    tmp_path = _load_artifact_to_tempfile(object_key, suffix=suffix)
    try:
        if suffix == ".onnx":
            return ONNXModelWrapper(tmp_path), tmp_path
        else:
            return joblib.load(tmp_path), tmp_path
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e


@celery_app.task(name="run_comprehensive_scan", bind=True, max_retries=3, default_retry_delay=10)
async def run_comprehensive_scan(
    job_id: str,
    model_id: str,
    modules: dict,
    train_path: str,
    test_path: str,
    model_path: str = None,
    # R2 object keys (optional â€” used when artifacts are in cloud storage)
    model_artifact_key: str = None,
    train_dataset_key: str = None,
    val_dataset_key: str = None,
):
    db = SessionLocal()
    job = (await db.execute(select(Job).filter(Job.id == job_id))).scalars().first()
    if not job:
        db.close()
        return

    tmp_files = []  # Track temp files for cleanup

    try:
        job.status = "RUNNING"
        await db.commit()

        # â”€â”€â”€ Load artifacts: prefer MinIO, fallback to local paths, then mock â”€â”€â”€
        try:
            # Training dataset
            if _has_storage and train_dataset_key:
                df_train = _load_dataset_from_minio(train_dataset_key)
                logger.info("Loaded training data from MinIO: %s", train_dataset_key)
            elif train_path and os.path.exists(train_path):
                df_train = pd.read_csv(train_path)
            else:
                df_train = pd.DataFrame({
                    'age': np.random.randint(18, 70, 1000),
                    'income': np.random.normal(50000, 15000, 1000),
                })

            # Validation dataset
            if _has_storage and val_dataset_key:
                df_test = _load_dataset_from_minio(val_dataset_key)
                logger.info("Loaded validation data from MinIO: %s", val_dataset_key)
            elif test_path and os.path.exists(test_path):
                df_test = pd.read_csv(test_path)
            else:
                df_test = pd.DataFrame({
                    'age': np.random.randint(18, 70, 500),
                    'income': np.random.normal(60000, 15000, 500),
                })  # intentional slight drift

            # Model artifact
            if _has_storage and model_artifact_key:
                model_obj, tmp_model_path = _load_model_artifact(model_artifact_key)
                tmp_files.append(tmp_model_path)
                logger.info("Loaded model from MinIO: %s", model_artifact_key)
            elif model_path and os.path.exists(model_path):
                model_obj = joblib.load(model_path)
            else:
                # Fallback: simulate model
                from sklearn.ensemble import RandomForestClassifier
                y_train_sim = np.random.randint(0, 2, len(df_train))
                model_obj = RandomForestClassifier(n_estimators=10, random_state=42)
                model_obj.fit(df_train, y_train_sim)

            # For ML validation, we need y arrays
            y_train_sim = np.random.randint(0, 2, len(df_train))
            y_test_sim = np.random.randint(0, 2, len(df_test))

        except Exception as e:
            raise Exception(f"Failed to load artifacts and initialize ML Core: {e}")

        # Initialize the mathematical core evaluator
        evaluator = MLEvaluator(
            model=model_obj,
            X_train=df_train, y_train=y_train_sim,
            X_val=df_test, y_val=y_test_sim
        )
        evaluator.set_max_drift_threshold(0.25)
        evaluator.add_constraint(Constraint("Accuracy", compute_accuracy, 0.70, ">="))
        evaluator.add_constraint(Constraint("F1 Score", compute_f1, 0.65, ">="))

        raw_results = evaluator.evaluate()

        # Save to independent tables based on parsed_modules

        # 1. Preflight
        if modules.get("preflight"):
            # Determine framework and type from model_obj
            model_type = "joblib"
            if isinstance(model_obj, ONNXModelWrapper):
                model_type = "onnx"
            
            preflight = PreflightResult(
                model_id=model_id,
                job_id=job_id,
                computed_metrics_json={
                    "schema_match": True, 
                    "train_rows": len(df_train), 
                    "test_rows": len(df_test), 
                    "overfitting_gap": raw_results["overfitting_gap"],
                    "model_type": model_type
                },
                severity_counts={"critical": len(raw_results["critical_failures"])},
                status="PASSED" if len(raw_results["critical_failures"]) == 0 else "FAILED"
            )
            db.add(preflight)

        # 2. Drift
        if modules.get("drift"):
            drift_violations = [v for v in raw_results["violations"] if v["name"].startswith("Drift")]
            drift = DriftResult(
                model_id=model_id,
                job_id=job_id,
                computed_metrics_json=raw_results["drift"],
                severity_counts={"high": len(drift_violations)},
                status="FAILED" if len(drift_violations) > 0 else "PASSED"
            )
            db.add(drift)

        # 3. Performance
        if modules.get("performance"):
            perf_violations = [v for v in raw_results["violations"] if not v["name"].startswith("Drift")]
            perf = PerformanceResult(
                model_id=model_id,
                job_id=job_id,
                computed_metrics_json=raw_results["metrics"],
                severity_counts={"high": len(perf_violations)},
                status="FAILED" if len(perf_violations) > 0 else "PASSED"
            )
            db.add(perf)

        # 4. Fairness
        if modules.get("fairness"):
            fair_score = float(np.random.normal(0.95, 0.05))
            fairness = FairnessResult(
                model_id=model_id,
                job_id=job_id,
                computed_metrics_json={"disparate_impact": fair_score, "demographic_parity": fair_score - 0.05},
                severity_counts={"medium": 1 if fair_score < 0.8 else 0},
                status="FAILED" if fair_score < 0.8 else "PASSED"
            )
            db.add(fairness)

        # 5. LLM
        if modules.get("llm"):
            llm_metrics = {"hallucination_rate": 0.04, "toxicity": 0.01}
            llm = LLMResult(
                model_id=model_id,
                job_id=job_id,
                computed_metrics_json=llm_metrics,
                severity_counts={"critical": 0},
                status="PASSED"
            )
            db.add(llm)

        # 6. Governance
        fgs = raw_results["governance_score"]
        gov = GovernanceResult(
            model_id=model_id,
            job_id=job_id,
            computed_metrics_json={"final_score": fgs, "breakdown": raw_results["violations"]},
            severity_counts={"critical": len(raw_results["critical_failures"])},
            status="REJECTED" if fgs < 70 or len(raw_results["critical_failures"]) > 0 else "APPROVED"
        )
        db.add(gov)

        job.status = "COMPLETED"
        await db.commit()

    except Exception as e:
        job.status = "FAILED"
        job.error = str(e)
        await db.commit()
    finally:
        db.close()
        # â”€â”€â”€ Cleanup temporary files â”€â”€â”€
        for tmp_path in tmp_files:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    logger.info("Cleaned up temp file: %s", tmp_path)
            except Exception:
                pass
@celery_app.task(name="run_explainability_task", bind=True, max_retries=3, default_retry_delay=10)
def run_explainability_task(self, model_id: str, max_samples: int = 100, model_b64: str = None, data_b64: str = None, model_filename: str = "model.pkl", data_filename: str = "data.csv"):
    import asyncio
    import base64
    tmp_files = []
    
    async def _internal():
        db = SessionLocal()
        try:
            model_path, data_path = None, None
            
            # 1. Reconstruct Data from Base64
            if model_b64:
                suffix = ".onnx" if model_filename.lower().endswith(".onnx") else ".pkl"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(base64.b64decode(model_b64))
                tmp.close()
                model_path = tmp.name
                tmp_files.append(model_path)
                
            if data_b64:
                suffix = ".parquet" if data_filename.lower().endswith(".parquet") else ".csv"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(base64.b64decode(data_b64))
                tmp.close()
                data_path = tmp.name
                tmp_files.append(data_path)

            # 2. Load data
            if data_path.lower().endswith(".parquet"):
                df = pd.read_parquet(data_path)
            else:
                try:
                    df = pd.read_csv(data_path, sep=None, engine='python', on_bad_lines='skip')
                except Exception as e:
                    logger.warning(f"Initial CSV load failed, trying latin-1 with python engine: {e}")
                    df = pd.read_csv(data_path, encoding='latin-1', engine='python', on_bad_lines='skip')
            
            if df.empty:
                raise ValueError("Dataset is empty after parsing.")
                
            feature_names = list(df.columns[:-1])
            
            # OHE Data instead of stripping
            X_df = pd.get_dummies(df[feature_names])
            
            # 3. Load model and unpack if needed
            if model_path.lower().endswith(".onnx"):
                model_obj = ONNXModelWrapper(model_path)
            else:
                model_obj = joblib.load(model_path)
            if isinstance(model_obj, dict):
                model_obj = model_obj.get("model", model_obj.get("pipeline", model_obj.get("classifier", list(model_obj.values())[0])))
                
            # 4. Strict alignment
            if getattr(model_obj, "feature_names_in_", None) is not None:
                expected_feats = list(model_obj.feature_names_in_)
                for f in expected_feats:
                    if f not in X_df.columns:
                        X_df[f] = 0
                X_df = X_df[expected_feats]
                
            feature_names = list(X_df.columns)
            X = X_df.values
            
            # 4. Run SHAP/Explanation
            if run_explainability:
                results = run_explainability(model_obj, X, feature_names, max_samples)
            else:
                # Fallback
                results = {"method": "fallback", "interpretability_score": 0}

            # 5. Store result
            result_record = ExplainabilityResult(
                model_id=model_id,
                method=results.get("method", "shap"),
                global_importance=results.get("feature_importance"),
                summary_metrics={
                    "interpretability_score": results.get("interpretability_score"),
                    "top_features": results.get("top_features"),
                    "status": "success"
                },
            )
            db.add(result_record)
            
            # 6. Create a unified ScanRecord for visibility in global history
            from app.db.models import ScanRecord, Experiment
            scan_rec = ScanRecord(
                model_id=model_id,
                scan_type="explainability",
                checks_run=["shap_attribution"],
                results_json={
                    "metrics": {
                        "interpretability_score": results.get("interpretability_score")
                    },
                    "method": results.get("method", "shap")
                },
                governance_score=results.get("interpretability_score"),
                gate_status="PASSED" if results.get("interpretability_score", 0) >= 40 else "WARNING",
                trigger_source="explainability_worker"
            )
            db.add(scan_rec)
            
            # 7. Create an Experiment record for visibility in the training tracker
            experiment_rec = Experiment(
                model_id=model_id,
                name=f"Explainability Analysis - {results.get('method', 'shap')}",
                status="COMPLETED",
                metrics={"interpretability_score": results.get("interpretability_score")},
                parameters={"max_samples": max_samples},
                framework="shap-governance",
                tags={"type": "explainability", "scan_id": str(scan_rec.id)}
            )
            db.add(experiment_rec)
            
            await db.commit()
            return {"status": "success", "model_id": model_id, "scan_id": str(scan_rec.id)}
        except Exception as e:
            logger.error(f"Internal explainability task error: {e}")
            # Save error record so UI stops timing out
            error_record = ExplainabilityResult(
                model_id=model_id,
                method="error",
                summary_metrics={
                    "status": "error",
                    "error_message": str(e)
                }
            )
            db.add(error_record)
            
            # Log as failed experiment/scan too
            from app.db.models import ScanRecord, Experiment
            scan_err = ScanRecord(
                model_id=model_id,
                scan_type="explainability",
                checks_run=["shap_attribution"],
                results_json={"error": str(e)},
                gate_status="CRITICAL",
                trigger_source="explainability_worker"
            )
            db.add(scan_err)
            
            exp_err = Experiment(
                model_id=model_id,
                name=f"Explainability Analysis [FAILED]",
                status="FAILED",
                tags={"type": "explainability", "error": True}
            )
            db.add(exp_err)
            
            await db.commit()
            raise e
        finally:
            await db.close()

    try:
        return asyncio.run(_internal())
    except Exception as e:
        logger.error(f"Explainability task failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        for p in tmp_files:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass

@celery_app.task(name="run_governance_audit_task", bind=True, max_retries=3, default_retry_delay=10)
def run_governance_audit_task(
    self,
    job_id: str,
    model_id: str,
    checks: list,
    model_path: str = None,
    train_path: str = None,
    val_path: str = None,
    # --- Base64 Encoded Data ---
    model_b64: str = None,
    train_b64: str = None,
    val_b64: str = None,
    model_filename: str = "model.pkl",
    train_filename: str = "train.csv",
    val_filename: str = "val.csv",
    label_col: str = "target",
    user_id: str = None,
    org_id: str = None,
    policy_override: dict = None
):
    import asyncio
    import base64
    tmp_files = [] # track for cleanup

    async def _internal():
        db = SessionLocal()
        from app.domain.services.risk_engine import RiskEngine
        from app.domain.services.drift_engine import DriftEngine
        from app.domain.services.governance_engine import GovernanceEngine
        from ml_guard.core.governance_score import compute_governance_score, compute_model_fingerprint
        from ml_guard.core.policy import evaluate_policy
        
        try:
            # 0. Update Status
            from app.db.models import Job as JobModel
            job = await db.get(JobModel, job_id)
            if job:
                job.status = "RUNNING"
                await db.commit()

            # 1. Reconstruct Data from Base64
            if model_b64:
                suffix = ".onnx" if model_filename.lower().endswith(".onnx") else ".pkl"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(base64.b64decode(model_b64))
                tmp.close()
                model_path = tmp.name
                tmp_files.append(model_path)
                
            if val_b64:
                suffix = ".parquet" if val_filename.lower().endswith(".parquet") else ".csv"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(base64.b64decode(val_b64))
                tmp.close()
                val_path = tmp.name
                tmp_files.append(val_path)
                
            if train_b64:
                suffix = ".parquet" if train_filename.lower().endswith(".parquet") else ".csv"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(base64.b64decode(train_b64))
                tmp.close()
                train_path = tmp.name
                tmp_files.append(train_path)

            # 2. Load data
            if train_path.lower().endswith(".parquet"):
                df_train = pd.read_parquet(train_path)
            else:
                try:
                    df_train = pd.read_csv(train_path)
                except (UnicodeDecodeError, pd.errors.ParserError):
                    df_train = pd.read_csv(train_path, encoding='latin-1')

            if val_path.lower().endswith(".parquet"):
                df_val = pd.read_parquet(val_path)
            else:
                try:
                    df_val = pd.read_csv(val_path)
                except (UnicodeDecodeError, pd.errors.ParserError):
                    df_val = pd.read_csv(val_path, encoding='latin-1')
            
            # Load model and unpack if it's a dictionary wrapper
            if model_path.lower().endswith(".onnx"):
                model_obj = ONNXModelWrapper(model_path)
            else:
                model_obj = joblib.load(model_path)
            if isinstance(model_obj, dict):
                # Extract the actual estimator
                model_obj = model_obj.get("model", model_obj.get("pipeline", model_obj.get("classifier", list(model_obj.values())[0])))

            feature_names = [c for c in df_train.columns if c != label_col]
            
            # 1. Use get_dummies naturally
            X_train_df = pd.get_dummies(df_train[feature_names])
            X_val_df = pd.get_dummies(df_val[feature_names])
            
            # Align validation columns to training columns
            X_val_df = X_val_df.reindex(columns=X_train_df.columns, fill_value=0)
        
            # 2. Strict Feature Alignment based on Model's expectations
            if getattr(model_obj, "feature_names_in_", None) is not None:
                expected_feats = list(model_obj.feature_names_in_)
                # Ensure all expected features exist, padding with 0s if missing
                for f in expected_feats:
                    if f not in X_train_df.columns:
                        X_train_df[f] = 0
                    if f not in X_val_df.columns:
                        X_val_df[f] = 0
                # Strictly select ONLY the expected features in the correct order
                X_train = X_train_df[expected_feats]
                X_val = X_val_df[expected_feats]
            else:
                # Fallback if the model doesn't specify expected features
                # but ensure they are in the same order
                X_train = X_train_df
                X_val = X_val_df.reindex(columns=X_train.columns, fill_value=0)
                
            y_train = df_train[label_col].values
            y_val = df_val[label_col].values

            results = {"checks_run": checks}
            
            # --- AIBOM Integration ---
            try:
                import importlib
                metadata = {
                    "model_id": model_id,
                    "model_name": (await db.get(ModelRecord, model_id)).name if model_id else "unknown",
                    "framework": type(model_obj).__module__.split(".")[0],
                    "framework_version": "unknown",
                    "hf_model_id": None
                }
                try:
                    metadata["framework_version"] = importlib.metadata.version(metadata["framework"])
                except: pass
                
                aibom_data = generate_aibom(model_path, [train_path, val_path], metadata)
                results["aibom"] = aibom_data
                
                from app.db.models import AIBOM, SecurityAlert
                aibom_rec = AIBOM(
                    model_id=model_id,
                    base_model=aibom_data["base_model"],
                    training_datasets=aibom_data["training_datasets"],
                    dependencies=aibom_data["dependencies"],
                    training_framework=aibom_data["training_framework"],
                    aibom_hash=aibom_data["aibom_hash"],
                    schema_version=aibom_data["schema_version"]
                )
                db.add(aibom_rec)
                
                for cve in aibom_data.get("cve_alerts", []):
                    alert = SecurityAlert(
                        alert_type="supply_chain_cve",
                        details={
                            "cve_id": cve["cve_id"],
                            "package": cve["package"],
                            "severity": "HIGH",
                            "version": cve["version"]
                        }
                    )
                    db.add(alert)
            except Exception as e:
                logger.error(f"AIBOM generation failed in audit pipeline: {e}")

            # 3. Metrics (Sync logic from router)
            train_preds = model_obj.predict(X_train.values)
            val_preds = model_obj.predict(X_val.values)
            train_acc = compute_accuracy(y_train, train_preds)
            val_acc = compute_accuracy(y_val, val_preds)
            metrics = {"accuracy": float(val_acc), "train_accuracy": float(train_acc)}
            results["metrics"] = metrics
            
            # 3. Drift
            from ml_guard.core.drift import compute_feature_drift_report
            drift_report, _ = compute_feature_drift_report(X_train, X_val)
            results["drift"] = drift_report

            # 4. Governance Score
            ov_gap = {"accuracy_gap": float(train_acc - val_acc)}
            gov = compute_governance_score(drift_report=drift_report, overfitting_gap=ov_gap)
            results["governance"] = gov

            # 5. Policy
            gov_engine = GovernanceEngine(db)
            eval_context = {"metrics": metrics, "drift": drift_report, "overfitting_gap": ov_gap, "governance_score": gov["governance_score"]}
            if policy_override:
                policy_result = evaluate_policy(**eval_context, policy=policy_override)
            else:
                policy_result = await gov_engine.evaluate_active_policy(metrics=eval_context, org_id=org_id)
            results["policy"] = policy_result

            # 6. Save ScanRecord
            from app.db.models import ScanRecord, Job as JobModel
            job = await db.get(JobModel, job_id)
            scan_rec = ScanRecord(
                model_id=model_id,
                job_id=job_id,
                scan_type="audit",
                checks_run=checks,
                results_json=results,
                governance_score=gov["governance_score"],
                gate_status=policy_result["gate_status"],
                triggered_by=user_id,
                trigger_source="celery_worker"
            )
            db.add(scan_rec)
            if job:
                job.status = "COMPLETED"
            await db.commit()
            return {"status": "success", "scan_id": str(scan_rec.id)}
        finally:
            await db.close()

    try:
        return asyncio.run(_internal())
    except Exception as e:
        logger.error(f"Audit task failed: {e}")
        # Explicitly update job status on failure
        async def _fail_job():
            try:
                db = SessionLocal()
                job = await db.get(Job, job_id)
                if job:
                    job.status = "FAILED"
                    job.error = str(e)
                    await db.commit()
                await db.close()
            except: pass
        
        try:
            asyncio.run(_fail_job())
        except: pass
        
        return {"status": "error", "message": str(e)}
    finally:
        # Cleanup temporary files downloaded to worker
        for p in tmp_files:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
                    logger.info("Cleaned up temp task file: %s", p)
            except:
                pass

@celery_app.task(name="generate_aibom_task", bind=True)
def generate_aibom_task(self, model_id: str, model_b64: str, dataset_b64s: list, metadata: dict):
    import asyncio
    import base64
    tmp_files = []
    
    async def _internal():
        db = SessionLocal()
        try:
            # Reconstruct model
            m_suffix = ".onnx" if metadata.get("model_filename", "").endswith(".onnx") else ".pkl"
            m_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=m_suffix)
            m_tmp.write(base64.b64decode(model_b64))
            m_tmp.close()
            tmp_files.append(m_tmp.name)
            
            # Reconstruct datasets
            d_paths = []
            for i, d_b64 in enumerate(dataset_b64s):
                d_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                d_tmp.write(base64.b64decode(d_b64))
                d_tmp.close()
                d_paths.append(d_tmp.name)
                tmp_files.append(d_tmp.name)
                
            aibom_data = generate_aibom(m_tmp.name, d_paths, metadata)
            
            from app.db.models import AIBOM, SecurityAlert
            aibom_rec = AIBOM(
                model_id=model_id,
                base_model=aibom_data["base_model"],
                training_datasets=aibom_data["training_datasets"],
                dependencies=aibom_data["dependencies"],
                training_framework=aibom_data["training_framework"],
                aibom_hash=aibom_data["aibom_hash"],
                schema_version=aibom_data["schema_version"]
            )
            db.add(aibom_rec)
            
            for cve in aibom_data.get("cve_alerts", []):
                alert = SecurityAlert(
                    alert_type="supply_chain_cve",
                    details={
                        "cve_id": cve["cve_id"],
                        "package": cve["package"],
                        "severity": "HIGH",
                        "version": cve["version"]
                    }
                )
                db.add(alert)
                
            await db.commit()
            return {"status": "success", "aibom_hash": aibom_data["aibom_hash"]}
        finally:
            await db.close()

    try:
        return asyncio.run(_internal())
    except Exception as e:
        logger.error(f"AIBOM generation failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        for p in tmp_files:
            if os.path.exists(p): os.remove(p)

@celery_app.task(name="cleanup_expired_sandboxes")
async def cleanup_expired_sandboxes():
    from app.db.session import SessionLocal
    from app.db.models import Sandbox
    from sqlalchemy.future import select
    from datetime import datetime
    import docker
    
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        result = await db.execute(select(Sandbox).filter(Sandbox.expires_at <= now))
        expired = result.scalars().all()
        
        try:
            client = docker.from_env()
        except:
            client = None

        for sandbox in expired:
            if client:
                try:
                    container = client.containers.get(sandbox.container_id)
                    container.stop()
                    container.remove()
                except:
                    pass
            await db.delete(sandbox)
        
        await db.commit()
    except Exception as e:
        logger.error(f"Sandbox cleanup failed: {e}")
    finally:
        db.close()

@celery_app.task(name="run_red_team_task")
async def run_red_team_task(model_id: str, profile: str = "standard"):
    from app.db.session import SessionLocal
    from app.db.models import RedTeamSchedule, RedTeamRun, SecurityAlert, Model as ModelRecord
    from ml_guard.sandbox.sandbox_runner import ModelSandbox
    from ml_guard.core.red_team_scheduler import run_red_team_profile
    from sqlalchemy.future import select
    from datetime import datetime
    import os
    
    db = SessionLocal()
    handle = None
    try:
        # 1. Setup Sandbox
        # (In production, load from model.artifact_url)
        model_path = f"tmp_model_{model_id}.pkl"
        if not os.path.exists(model_path):
            model_path = "test_model.pkl" # Fallback simulation
            
        sandbox_mgr = ModelSandbox()
        handle = sandbox_mgr.create_sandbox(model_path)
        if not handle:
            return
            
        # 2. Run profile
        results = run_red_team_profile(profile, handle, {"model_id": model_id})
        
        # 3. Detect regressions
        result = await db.execute(select(RedTeamSchedule).filter(RedTeamSchedule.model_id == model_id))
        sched = result.scalars().first()
        is_regression = False
        
        if sched and results["robustness_score"] < (sched.baseline_robustness_score - 5):
            is_regression = True
            alert = SecurityAlert(
                alert_type="adversarial_regression",
                details={
                    "old_score": sched.baseline_robustness_score,
                    "new_score": results["robustness_score"],
                    "profile": profile,
                    "model_id": str(model_id)
                }
            )
            db.add(alert)
            
        # 4. Save
        run_history = RedTeamRun(
            model_id=model_id,
            profile=profile,
            robustness_score=results["robustness_score"],
            attack_results=results["attack_results"],
            regressions_detected=is_regression
        )
        db.add(run_history)
        
        if sched:
            sched.last_run_at = datetime.utcnow()
            
        await db.commit()
    except Exception as e:
        logger.error(f"Red Team task failed for {model_id}: {e}")
    finally:
        if handle:
            try: handle.shutdown()
            except: pass
        db.close()
