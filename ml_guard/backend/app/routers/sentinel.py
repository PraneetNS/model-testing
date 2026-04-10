from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import Dict, List, Any
import json
import time
import os
import httpx
import structlog
from app.db.session import SessionLocal
from app.db.models import SentinelRecord, Model, AlertRule
from app.core.config import settings

router = APIRouter()
logger = structlog.get_logger()

# Dashboard Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, model_id: str, websocket: WebSocket):
        await websocket.accept()
        if model_id not in self.active_connections:
            self.active_connections[model_id] = []
        self.active_connections[model_id].append(websocket)

    def disconnect(self, model_id: str, websocket: WebSocket):
        if model_id in self.active_connections:
            self.active_connections[model_id].remove(websocket)

    async def broadcast(self, model_id: str, message: dict):
        if model_id in self.active_connections:
            for connection in self.active_connections[model_id]:
                await connection.send_json(message)

manager = ConnectionManager()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.websocket("/stream/{model_id}")
async def sentinel_stream(websocket: WebSocket, model_id: str):
    """
    WebSocket endpoint for Sidecar Agents to stream real-time PSI deltas.
    """
    await websocket.accept()
    logger.info("Sentinel agent connected", model_id=model_id)
    
    db = SessionLocal()
    try:
        while True:
            data = await websocket.receive_json()
            avg_psi = data.get("avg_psi", 0.0)
            
            # 1. Persist to DB
            record = SentinelRecord(
                model_id=model_id,
                avg_psi=avg_psi,
                feature_psi=data.get("feature_psi"),
                window_size=data.get("window_size"),
                threshold=0.2, # Default threshold
                is_breached=avg_psi > 0.2
            )
            db.add(record)
            await db.commit()
            
            # 2. Broadcast to Dashboards
            await manager.broadcast(model_id, {
                "type": "SENTINEL_UPDATE",
                "model_id": model_id,
                "avg_psi": avg_psi,
                "timestamp": time.time(),
                "is_breached": avg_psi > 0.2
            })
            
            # 3. Threshold Evaluation & Webhook
            if avg_psi > 0.2:
                await _trigger_webhook(model_id, avg_psi)
                
    except WebSocketDisconnect:
        logger.info("Sentinel agent disconnected", model_id=model_id)
    finally:
        db.close()

@router.websocket("/live/ws/{model_id}")
async def dashboard_live_stream(websocket: WebSocket, model_id: str):
    """
    WebSocket endpoint for dashboards to receive live sentinel updates.
    """
    await manager.connect(model_id, websocket)
    try:
        while True:
            await websocket.receive_text() # Keep alive
    except WebSocketDisconnect:
        manager.disconnect(model_id, websocket)

@router.get("/{model_id}/live")
async def get_live_sentinel_data(model_id: str, db: AsyncSession = Depends(get_db)):
    """Return last 100 PSI points for historical plotting."""
    records = db.query(SentinelRecord).filter(
        SentinelRecord.model_id == model_id
    ).order_by(SentinelRecord.created_at.desc()).limit(100).all()
    
    return [
        {
            "timestamp": r.created_at.isoformat(),
            "avg_psi": r.avg_psi,
            "is_breached": r.is_breached
        } for r in reversed(records)
    ]

async def _trigger_webhook(model_id: str, psi: float):
    """Dispatch Slack/Generic Webhook on drift breach."""
    logger.warning("Sentinel drift detected", model_id=model_id, psi=psi)
    
    # In a real app, pull webhook URL from AlertRule registry
    webhook_url = os.getenv("SENTINEL_WEBHOOK_URL")
    if not webhook_url:
        return

    payload = {
        "text": f"🚨 *ML Guard Sentinel Alert* 🚨\nModel `{model_id}` is experiencing high drift!\n*Current PSI:* {psi:.4f}\n*Threshold:* 0.2000\n*Action:* Rollback suggested.",
        "model_id": model_id,
        "metric": "psi",
        "value": psi,
        "rollback_suggested": True
    }
    
    async with httpx.AsyncClient() as client:
        try:
            await client.post(webhook_url, json=payload)
        except Exception as e:
            logger.error("Failed to send sentinel webhook", error=str(e))
