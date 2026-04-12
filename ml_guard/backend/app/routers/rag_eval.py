import uuid
import csv
import io
import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse

from app.db.session import get_db
from app.db.models import RagTrace, AlertEvent
from ml_guard.core.rag import context_relevance, grounding_fidelity, retrieval_hit_rate, hallucination_risk

router = APIRouter()

class RagLogRequest(BaseModel):
    query: str
    retrieved_chunks: List[str]
    retrieved_doc_ids: List[str]
    answer: str
    latency_ms: Optional[float] = None

@router.post("/rag-eval/{model_id}/log")
async def log_rag_trace(model_id: str, payload: RagLogRequest, db: AsyncSession = Depends(get_db)):
    c_rel = context_relevance(payload.query, payload.retrieved_chunks)
    g_fid = grounding_fidelity(payload.answer, payload.retrieved_chunks)
    h_risk = hallucination_risk(payload.answer, payload.retrieved_chunks)
    
    trace = RagTrace(
        model_id=model_id,
        query=payload.query,
        answer=payload.answer,
        retrieved_chunks=payload.retrieved_chunks,
        retrieved_doc_ids=payload.retrieved_doc_ids,
        latency_ms=payload.latency_ms,
        context_relevance=c_rel,
        grounding_fidelity=g_fid,
        hallucination_risk=h_risk
    )
    db.add(trace)
    
    await db.commit()
    
    one_hour_ago = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
    
    recent_traces = (await db.execute(
        select(RagTrace)
        .filter(RagTrace.model_id == model_id, RagTrace.timestamp >= one_hour_ago)
    )).scalars().all()
    
    num_traces = len(recent_traces)
    if num_traces > 0:
        high_risk_traces = sum(1 for t in recent_traces if t.hallucination_risk == "high")
        if (high_risk_traces / num_traces) > 0.2:
            alert = AlertEvent(
                rule_id=str(uuid.uuid4()), 
                scan_id=str(uuid.uuid4()), 
                severity="HIGH",
                message=f"RAG Hallucination Risk spiked: {high_risk_traces}/{num_traces} high risk traces in last hour."
            )
            db.add(alert)
            await db.commit()
    
    return {"status": "logged", "trace_id": str(trace.id), "hallucination_risk": h_risk}

@router.get("/rag-eval/{model_id}/report")
async def get_rag_report(model_id: str, limit: int = 100, db: AsyncSession = Depends(get_db)):
    traces = (await db.execute(
        select(RagTrace)
        .filter(RagTrace.model_id == model_id)
        .order_by(RagTrace.timestamp.desc())
        .limit(limit)
    )).scalars().all()
    
    if not traces:
        return {"error": "No traces found"}
        
    avg_context_relevance = sum(t.context_relevance for t in traces if t.context_relevance is not None) / len(traces)
    avg_grounding_fidelity = sum(t.grounding_fidelity for t in traces if t.grounding_fidelity is not None) / len(traces)
    
    risks = {"low": 0, "medium": 0, "high": 0}
    for t in traces:
        if t.hallucination_risk in risks:
            risks[t.hallucination_risk] += 1
            
    p95_latency = 0.0
    latencies = sorted([t.latency_ms for t in traces if t.latency_ms is not None])
    if latencies:
        idx = int(len(latencies) * 0.95)
        p95_latency = latencies[idx] if idx < len(latencies) else latencies[-1]
        
    return {
        "avg_context_relevance": avg_context_relevance,
        "avg_grounding_fidelity": avg_grounding_fidelity,
        "retrieval_hit_rate": 0.0, 
        "hallucination_risk_distribution": risks,
        "p95_latency_ms": p95_latency,
        "time_series": [{"time": t.timestamp.isoformat(), "grounding_fidelity": t.grounding_fidelity} for t in sorted(traces, key=lambda x: x.timestamp)]
    }

@router.post("/rag-eval/{model_id}/evaluate-batch")
async def evaluate_batch(model_id: str, file: UploadFile = File(...)):
    import json
    content = await file.read()
    lines = content.decode("utf-8").strip().split('\n')
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["query", "context_relevance", "grounding_fidelity", "hallucination_risk"])
    
    for line in lines:
        if not line.strip(): continue
        data = json.loads(line)
        q = data.get("query", "")
        chunks = data.get("chunks", [])
        ans = data.get("answer", "")
        
        c_rel = context_relevance(q, chunks)
        g_fid = grounding_fidelity(ans, chunks)
        h_risk = hallucination_risk(ans, chunks)
        
        writer.writerow([q, c_rel, g_fid, h_risk])
        
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=rag_eval_{model_id}.csv"}
    )
