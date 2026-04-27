from fastapi import Request, HTTPException, Depends
from sqlalchemy.future import select
from sqlalchemy import func
from datetime import datetime, timezone
from app.db.session import AsyncSessionLocal
from app.db.models import Organization, SubscriptionPlan, UsageEvent
import structlog

logger = structlog.get_logger()

async def check_billing_limits(request: Request):
    """
    Middleware-style dependency to enforce plan limits.
    Returns 402 Payment Required if limits are exceeded.
    """
    # 1. Identify the metered event based on the endpoint path
    path = request.url.path
    event_type = None
    
    if "/audit" in path: 
        event_type = "model_audited"
    elif "/reports" in path: 
        event_type = "governance_report_generated"
    elif "/compliance" in path: 
        event_type = "compliance_pack_run"
    elif "/guardrail" in path: 
        event_type = "guardrail_evaluated"
    elif "/redteam" in path:
        event_type = "red_team_run"
    elif "/aibom" in path:
        event_type = "aibom_generated"

    if not event_type:
        return

    # 2. Identify the organization
    x_org_id = request.headers.get("X-Org-ID")
    
    async with AsyncSessionLocal() as db:
        if not x_org_id:
            # For development/demo purposes, pick the first org
            org_stmt = select(Organization).limit(1)
            org = (await db.execute(org_stmt)).scalars().first()
            if not org: return
            org_id = org.id
        else:
            org_id = x_org_id

        # 3. Fetch Org and Plan limits
        org_stmt = select(Organization).filter(Organization.id == org_id)
        org = (await db.execute(org_stmt)).scalars().first()
        if not org: return

        plan_id = org.plan_id
        if not plan_id:
            # Default to free plan if none set
            free_plan = (await db.execute(select(SubscriptionPlan).filter(SubscriptionPlan.slug == "free"))).scalars().first()
            plan = free_plan
        else:
            plan = (await db.execute(select(SubscriptionPlan).filter(SubscriptionPlan.id == plan_id))).scalars().first()

        if not plan: return

        # 4. Determine limit for this event type
        limit = -1
        if event_type == "governance_report_generated":
            limit = plan.reports_per_month
        elif event_type == "compliance_pack_run":
            limit = plan.compliance_packs_limit
        elif event_type == "guardrail_evaluated":
            limit = plan.guardrail_eval_limit
        elif event_type == "model_audited":
            # For demo, audits are unlimited on pro/enterprise but 10 on free
            limit = 10 if plan.slug == "free" else -1
            
        if limit == -1:
            return # Unlimited

        # 5. Check current month's usage
        now = datetime.now(timezone.utc)
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        usage_stmt = select(func.sum(UsageEvent.quantity)).filter(
            UsageEvent.org_id == org_id,
            UsageEvent.event_type == event_type,
            UsageEvent.timestamp >= start
        )
        used = (await db.execute(usage_stmt)).scalar() or 0
        
        if used >= limit:
            logger.warning("USAGE_LIMIT_EXCEEDED", org_id=org_id, event_type=event_type, limit=limit)
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "usage_limit_reached",
                    "event_type": event_type,
                    "limit": limit,
                    "upgrade_url": "https://niyantrana.ai/pricing"
                }
            )
