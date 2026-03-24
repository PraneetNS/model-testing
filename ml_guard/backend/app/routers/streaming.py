"""
Real-Time Streaming Drift Detection Engine.

WebSocket endpoint: /api/v1/stream/production
- Maintains rolling window of last N=1000 predictions
- Computes rolling PSI, JSD, calibration, stability
- Triggers alerts when thresholds breached
- Persists rolling metrics to DB

Protocol:
  Client sends JSON: {"features": {...}, "prediction": float, "confidence": float, "actual": float|null}
  Server pushes back: {"type": "metrics", "rolling_psi": ..., "rolling_jsd": ..., ...}
  Server pushes:      {"type": "alert", "metric": ..., "value": ..., "threshold": ...}
"""
import json
import time
import uuid
import asyncio
import hashlib
import numpy as np
from collections import deque
from typing import Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.db.models import ScanRecord, AlertEvent, AlertRule, User, Organization, AuditLog
from app.core.auth import AuthContext, log_action

router = APIRouter()

# ─── Rolling Window Store (per model) ───
class RollingWindow:
    """In-memory sliding window for streaming metrics."""
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.predictions = deque(maxlen=max_size)
        self.confidences = deque(maxlen=max_size)
        self.actuals = deque(maxlen=max_size)
        self.features: deque = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self._baseline_hist: Optional[np.ndarray] = None
        self._baseline_bins: Optional[np.ndarray] = None

    def add(self, prediction: float, confidence: float = None,
            actual: float = None, feature_vec: dict = None):
        self.predictions.append(prediction)
        self.confidences.append(confidence or 0.5)
        self.actuals.append(actual)
        self.features.append(feature_vec or {})
        self.timestamps.append(time.time())

    def set_baseline(self, baseline_preds: list):
        """Establish baseline distribution from reference data."""
        arr = np.array(baseline_preds, dtype=float)
        self._baseline_hist, self._baseline_bins = np.histogram(arr, bins=20, density=True)
        self._baseline_hist = self._baseline_hist + 1e-10  # avoid zero

    @property
    def count(self) -> int:
        return len(self.predictions)

    def compute_rolling_psi(self) -> Optional[float]:
        if self._baseline_hist is None or self.count < 50:
            return None
        current = np.array(list(self.predictions), dtype=float)
        current_hist, _ = np.histogram(current, bins=self._baseline_bins, density=True)
        current_hist = current_hist + 1e-10
        baseline = self._baseline_hist
        psi = float(np.sum((current_hist - baseline) * np.log(current_hist / baseline)))
        return round(abs(psi), 6)

    def compute_rolling_jsd(self) -> Optional[float]:
        if self._baseline_hist is None or self.count < 50:
            return None
        current = np.array(list(self.predictions), dtype=float)
        current_hist, _ = np.histogram(current, bins=self._baseline_bins, density=True)
        current_hist = current_hist + 1e-10
        p = self._baseline_hist / self._baseline_hist.sum()
        q = current_hist / current_hist.sum()
        m = 0.5 * (p + q)
        jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        return round(float(jsd), 6)

    def compute_rolling_calibration(self) -> Optional[dict]:
        """Brier score on recent predictions where actuals are available."""
        pairs = [(c, a) for c, a in zip(self.confidences, self.actuals) if a is not None]
        if len(pairs) < 20:
            return None
        confs, acts = zip(*pairs)
        confs = np.array(confs, dtype=float)
        acts = np.array(acts, dtype=float)
        brier = float(np.mean((confs - acts) ** 2))
        return {"brier_score": round(brier, 6), "n_labeled": len(pairs)}

    def compute_rolling_stability(self) -> Optional[dict]:
        """Variance-based stability of recent predictions."""
        if self.count < 50:
            return None
        recent = np.array(list(self.predictions)[-100:], dtype=float)
        variance = float(np.var(recent))
        mean = float(np.mean(recent))
        cv = float(np.std(recent) / mean) if mean != 0 else 0
        # Compare first half vs second half
        half = len(recent) // 2
        first_half_var = float(np.var(recent[:half]))
        second_half_var = float(np.var(recent[half:]))
        stability_score = 1.0 - min(1.0, abs(second_half_var - first_half_var) / max(first_half_var, 1e-10))
        return {
            "variance": round(variance, 6),
            "cv": round(cv, 4),
            "stability_score": round(stability_score, 4),
        }

    def get_snapshot(self) -> dict:
        return {
            "window_size": self.count,
            "rolling_psi": self.compute_rolling_psi(),
            "rolling_jsd": self.compute_rolling_jsd(),
            "rolling_calibration": self.compute_rolling_calibration(),
            "rolling_stability": self.compute_rolling_stability(),
            "latest_prediction": float(self.predictions[-1]) if self.predictions else None,
            "mean_confidence": round(float(np.mean(list(self.confidences))), 4) if self.confidences else None,
        }


# ─── Per-model window registry ───
_windows: Dict[str, RollingWindow] = {}

def _get_window(model_id: str) -> RollingWindow:
    if model_id not in _windows:
        _windows[model_id] = RollingWindow(max_size=1000)
    return _windows[model_id]


# ─── Default thresholds ───
STREAM_THRESHOLDS = {
    "max_psi": 0.25,
    "max_jsd": 0.15,
    "min_stability": 0.80,
    "max_brier": 0.30,
}


def _check_alerts(snapshot: dict, thresholds: dict = None) -> list:
    """Check snapshot against thresholds, return alert list."""
    th = thresholds or STREAM_THRESHOLDS
    alerts = []
    psi = snapshot.get("rolling_psi")
    if psi is not None and psi > th["max_psi"]:
        alerts.append({"metric": "rolling_psi", "value": psi, "threshold": th["max_psi"], "severity": "CRITICAL"})
    jsd = snapshot.get("rolling_jsd")
    if jsd is not None and jsd > th["max_jsd"]:
        alerts.append({"metric": "rolling_jsd", "value": jsd, "threshold": th["max_jsd"], "severity": "WARNING"})
    stab = snapshot.get("rolling_stability")
    if stab and stab.get("stability_score", 1) < th["min_stability"]:
        alerts.append({"metric": "stability_score", "value": stab["stability_score"], "threshold": th["min_stability"], "severity": "CRITICAL"})
    cal = snapshot.get("rolling_calibration")
    if cal and cal.get("brier_score", 0) > th["max_brier"]:
        alerts.append({"metric": "brier_score", "value": cal["brier_score"], "threshold": th["max_brier"], "severity": "WARNING"})
    return alerts


# ═══════════════════════════════════════════════
# WEBSOCKET ENDPOINT
# ═══════════════════════════════════════════════
@router.websocket("/ws/stream")
async def stream_production(websocket: WebSocket, model_id: str = Query("default")):
    """
    Real-time streaming drift detection via WebSocket.

    Client sends prediction events, server returns rolling metrics + alerts.
    """
    await websocket.accept()
    window = _get_window(model_id)
    metrics_interval = 10  # send metrics every N predictions
    counter = 0

    try:
        # Resolve a mock context for the websocket (since WS doesn't easily use Depends(require_role))
        db_auth = SessionLocal()
        org = db_auth.query(Organization).first()
        user = db_auth.query(User).filter(User.org_id == org.id).first() if org else None
        auth = AuthContext(user=user, org=org) if user and org else None
        db_auth.close()

        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            # Handle baseline setup
            if data.get("type") == "baseline":
                baseline_preds = data.get("predictions", [])
                if baseline_preds:
                    window.set_baseline(baseline_preds)
                    await websocket.send_json({"type": "baseline_set", "n": len(baseline_preds)})
                continue

            # Handle prediction event
            prediction = data.get("prediction")
            if prediction is None:
                await websocket.send_json({"type": "error", "message": "Missing 'prediction' field"})
                continue

            window.add(
                prediction=float(prediction),
                confidence=float(data.get("confidence", 0.5)),
                actual=float(data["actual"]) if data.get("actual") is not None else None,
                feature_vec=data.get("features"),
            )
            counter += 1

            # Send ack
            await websocket.send_json({"type": "ack", "window_size": window.count})

            # Periodic metrics push
            if counter % metrics_interval == 0:
                snapshot = window.get_snapshot()
                await websocket.send_json({"type": "metrics", **snapshot})

                # Check alerts
                alerts = _check_alerts(snapshot)
                for alert in alerts:
                    await websocket.send_json({"type": "alert", **alert})

                # Persist to DB periodically (every 100 events)
                if counter % 100 == 0:
                    try:
                        db = SessionLocal()
                        scan_rec = ScanRecord(
                            model_id=model_id if model_id != "default" else None,
                            scan_type="stream",
                            checks_run=["rolling_psi", "rolling_jsd", "rolling_calibration", "rolling_stability"],
                            results_json=snapshot,
                            governance_score=None,
                            gate_status="MONITORING",
                            trigger_source="stream",
                        )
                        db.add(scan_rec)
                        db.commit()
                        if auth:
                            log_action(db, auth, "stream.persist", resource_type="model", resource_id=model_id, details={"window_size": window.count})
                        db.close()
                    except Exception:
                        pass

    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close()


# ═══════════════════════════════════════════════
# HTTP ENDPOINTS FOR STREAMING STATE
# ═══════════════════════════════════════════════
@router.get("/stream/status/{model_id}")
async def stream_status(model_id: str):
    """Get current rolling window metrics for a model."""
    if model_id not in _windows:
        return {"model_id": model_id, "active": False, "message": "No streaming data for this model."}
    window = _windows[model_id]
    snapshot = window.get_snapshot()
    alerts = _check_alerts(snapshot)
    return {
        "model_id": model_id,
        "active": True,
        **snapshot,
        "active_alerts": alerts,
    }


@router.get("/stream/models")
async def list_streaming_models():
    """List all models with active streaming windows."""
    return {
        "streaming_models": [
            {"model_id": mid, "window_size": w.count, "has_baseline": w._baseline_hist is not None}
            for mid, w in _windows.items()
        ]
    }


@router.post("/stream/baseline/{model_id}")
async def set_baseline(model_id: str, predictions: list):
    """Set baseline distribution for a model via HTTP (alternative to WebSocket)."""
    window = _get_window(model_id)
    window.set_baseline(predictions)
    return {"model_id": model_id, "baseline_size": len(predictions), "status": "set"}


@router.post("/stream/production")
async def post_stream_prediction(model_id: str = Query("default"), data: dict = Body(...)):
    """Bridge for REST-based streaming ingestion."""
    window = _get_window(model_id)
    prediction = data.get("prediction")
    if prediction is None:
        raise HTTPException(status_code=400, detail="Missing 'prediction'")

    window.add(
        prediction=float(prediction),
        confidence=float(data.get("confidence", 0.5)),
        actual=float(data["actual"]) if data.get("actual") is not None else None,
        feature_vec=data.get("features"),
    )
    return {"status": "accepted", "window_size": window.count}
