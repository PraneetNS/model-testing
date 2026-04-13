import hashlib
import uuid
import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from app.db.session import get_db
from app.db.models import Agent, AgentTrace
from app.core.auth import AuthContext, require_role
from ml_guard.core.agent_auditor import audit_step, compute_step_risk

router = APIRouter()

@router.post("/agent-eval/register")
async def register_agent(
    payload: Dict = Body(...),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer"))
):
    # registers a new agent: {name, allowed_tools: [], sensitive_topics: [], step_sla_ms, owner_key_id}
    agent = Agent(
        id=uuid.uuid4(),
        name=payload["name"],
        allowed_tools=payload.get("allowed_tools", []),
        sensitive_topics=payload.get("sensitive_topics", []),
        step_sla_ms=payload.get("step_sla_ms", 5000),
        owner_key_id=uuid.UUID(payload["owner_key_id"]) if "owner_key_id" in payload else None
    )
    db.add(agent)
    await db.commit()
    return {"agent_id": str(agent.id), "status": "registered"}

@router.post("/agent-eval/trace")
async def ingest_trace(
    payload: Dict = Body(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Ingests agent traces, runs analysis, and stores hashed manifests.
    Accepts: {agent_id, session_id, steps: [{type, input, output, tool, latency_ms}]}
    """
    agent_id = payload.get("agent_id")
    result = await db.execute(select(Agent).filter(Agent.id == agent_id))
    agent = result.scalars().first()
    if not agent:
        raise HTTPException(404, "Agent not found")

    session_id = payload.get("session_id")
    steps = payload.get("steps", [])
    
    # Fetch existing session history for loop detection
    h_result = await db.execute(
        select(AgentTrace).filter(AgentTrace.session_id == session_id).order_by(AgentTrace.trace_index)
    )
    history = h_result.scalars().all()
    
    ingested_traces = []
    current_index = len(history)
    
    for step in steps:
        # a. Audit step before hashing (needs raw content)
        violations = audit_step(step, agent, history + ingested_traces)
        risk_score = compute_step_risk(violations)
        
        # b. Hash summaries (PII protection)
        input_hash = hashlib.sha256(str(step.get("input", "")).encode()).hexdigest()
        output_hash = hashlib.sha256(str(step.get("output", "")).encode()).hexdigest()
        
        trace = AgentTrace(
            id=uuid.uuid4(),
            agent_id=agent_id,
            session_id=session_id,
            trace_index=current_index,
            step_type=step["type"],
            input_summary=input_hash,
            output_summary=output_hash,
            tool_name=step.get("tool"),
            latency_ms=step.get("latency_ms", 0),
            policy_violations=violations,
            risk_score=risk_score,
            flagged=len(violations) > 0
        )
        db.add(trace)
        ingested_traces.append(trace)
        current_index += 1
        
    await db.commit()
    return {"status": "ingested", "count": len(ingested_traces)}

@router.get("/agent-eval/{agent_id}/report")
async def get_agent_report(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer"))
):
    # Returns: {total_sessions, total_steps, violation_rate, top_violations, avg_latency_ms, risk_score_trend}
    
    # Basic metrics
    m_result = await db.execute(
        select(
            func.count(AgentTrace.id),
            func.avg(AgentTrace.latency_ms),
            func.count(AgentTrace.session_id.distinct())
        ).filter(AgentTrace.agent_id == agent_id)
    )
    total_steps, avg_latency, total_sessions = m_result.one()
    
    if total_steps == 0:
        return {"total_sessions": 0, "total_steps": 0, "message": "No data available"}

    # Violation rate
    v_result = await db.execute(
        select(func.count(AgentTrace.id)).filter(AgentTrace.agent_id == agent_id, AgentTrace.flagged == True)
    )
    flagged_steps = v_result.scalar()
    
    return {
        "total_sessions": total_sessions,
        "total_steps": total_steps,
        "violation_rate": flagged_steps / total_steps,
        "avg_latency_ms": float(avg_latency or 0),
        "risk_score_trend": [] # Placeholder for 7-day trend
    }

@router.get("/agent-eval/{agent_id}/sessions/{session_id}")
async def get_session_replay(
    agent_id: str,
    session_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer"))
):
    result = await db.execute(
        select(AgentTrace).filter(AgentTrace.session_id == session_id).order_by(AgentTrace.trace_index)
    )
    traces = result.scalars().all()
    return [
        {
            "index": t.trace_index,
            "type": t.step_type,
            "tool": t.tool_name,
            "latency": t.latency_ms,
            "violations": t.policy_violations,
            "risk_score": t.risk_score,
            "hashed_io": {"input": t.input_summary, "output": t.output_summary}
        } for t in traces
    ]

@router.get("/agent-eval/{agent_id}/graph")
async def get_agent_dag(
    agent_id: str,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(AgentTrace).filter(AgentTrace.agent_id == agent_id).order_by(AgentTrace.session_id, AgentTrace.trace_index)
    )
    all_traces = result.scalars().all()
    
    nodes = {}
    edges = {}
    prev_node = None
    prev_session = None
    
    for t in all_traces:
        node_key = f"{t.step_type}:{t.tool_name or ''}"
        nodes[node_key] = nodes.get(node_key, 0) + 1
        
        if prev_session == t.session_id:
            edge_key = (prev_node, node_key)
            edges[edge_key] = edges.get(edge_key, 0) + 1
            
        prev_node = node_key
        prev_session = t.session_id
        
    return {
        "nodes": [{"id": k, "count": v} for k, v in nodes.items()],
        "edges": [{"from": k[0], "to": k[1], "count": v} for k, v in edges.items()]
    }
