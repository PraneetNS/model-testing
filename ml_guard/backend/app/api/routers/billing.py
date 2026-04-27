from fastapi import APIRouter, Depends, HTTPException, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, extract
from typing import List, Optional
import stripe
from datetime import datetime, timezone, timedelta

from app.db.session import get_db
from app.db.models import Organization, SubscriptionPlan, UsageEvent, Model
from app.schemas.billing import BillingUsageResponse, UsageStat, SubscriptionDetails, CheckoutResponse, SubscribeResponse
from app.billing.stripe_client import StripeClient
from app.core.config import settings
import structlog

logger = structlog.get_logger()
router = APIRouter()

# Current month helper
def get_month_range():
    now = datetime.now(timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    # Next month start
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end

@router.get("/usage", response_model=BillingUsageResponse)
async def get_usage(
    db: AsyncSession = Depends(get_db),
    # In a real app, we'd get org_id from the authenticated user
    # For now, we'll assume a default org or header for demo purposes
    x_org_id: Optional[str] = Header(None)
):
    if not x_org_id:
        # Fallback to first org for demo
        org = (await db.execute(select(Organization))).scalars().first()
        if not org: raise HTTPException(status_code=404, detail="No organization found")
        org_id = org.id
    else:
        org_id = x_org_id

    org = (await db.execute(select(Organization).filter(Organization.id == org_id))).scalars().first()
    plan = (await db.execute(select(SubscriptionPlan).filter(SubscriptionPlan.id == org.plan_id))).scalars().first()
    if not plan:
        # Fallback to free plan
        plan = (await db.execute(select(SubscriptionPlan).filter(SubscriptionPlan.slug == "free"))).scalars().first()

    start, end = get_month_range()
    
    # Query aggregated usage
    usage_stmt = select(UsageEvent.event_type, func.sum(UsageEvent.quantity)).filter(
        UsageEvent.org_id == org_id,
        UsageEvent.timestamp >= start,
        UsageEvent.timestamp < end
    ).group_by(UsageEvent.event_type)
    
    usage_results = (await db.execute(usage_stmt)).all()
    usage_map = {row[0]: row[1] for row in usage_results}

    # Define limits and overage pricing
    # predictions over limit: $0.0001 each
    # compliance_certificate: $500 each (one-time) - handled via checkout
    # red_team_run: $25 each (standard profile), $100 (exhaustive)
    
    metrics = [
        ("prediction_logged", plan.predictions_per_month, 0.0001),
        ("governance_report_generated", plan.reports_per_month, 10.0),
        ("compliance_pack_run", plan.compliance_packs_limit, 50.0),
        ("guardrail_evaluated", plan.guardrail_eval_limit, 0.01),
        ("model_audited", -1, 0), # No explicit limit mentioned for audit in plan seed but enterprise is unlimited
    ]

    stats = []
    for event_type, limit, overage_rate in metrics:
        used = usage_map.get(event_type, 0)
        limit_val = limit if limit != -1 else 999999999
        pct = (used / limit_val * 100) if limit_val > 0 else 0
        overage = max(0, used - limit_val) if limit != -1 else 0
        cost = overage * overage_rate
        
        stats.append(UsageStat(
            event_type=event_type,
            used=used,
            limit=limit,
            pct_used=min(100.0, pct),
            overage_units=overage,
            overage_cost_usd=cost
        ))

    return BillingUsageResponse(
        organization_id=org_id,
        plan_name=plan.name,
        usage=stats
    )

@router.post("/subscribe/{plan_slug}", response_model=SubscribeResponse)
async def subscribe(
    plan_slug: str,
    db: AsyncSession = Depends(get_db),
    x_org_id: Optional[str] = Header(None)
):
    # 1. Get Org and User info
    org = (await db.execute(select(Organization).filter(Organization.id == x_org_id))).scalars().first()
    if not org: raise HTTPException(status_code=404, detail="Org not found")
    
    plan = (await db.execute(select(SubscriptionPlan).filter(SubscriptionPlan.slug == plan_slug))).scalars().first()
    if not plan: raise HTTPException(status_code=404, detail="Plan not found")

    # 2. Ensure Stripe Customer exists
    if not org.stripe_customer_id:
        # In a real app, we'd use the admin's email
        customer_id = StripeClient.create_customer(str(org.id), f"admin@{org.slug}.com", org.name)
        org.stripe_customer_id = customer_id
        await db.commit()

    # 3. Create Subscription (Mocked Price ID for demo)
    # In reality, you'd have a mapping of plan_slug -> stripe_price_id
    price_id = f"price_{plan_slug}_monthly" 
    sub_data = StripeClient.create_subscription(org.stripe_customer_id, price_id)
    
    org.stripe_subscription_id = sub_data["id"]
    org.plan_id = plan.id
    await db.commit()

    return SubscribeResponse(
        subscription_id=sub_data["id"],
        client_secret=sub_data["client_secret"]
    )

@router.post("/certificate-checkout/{model_id}/{pack_name}", response_model=CheckoutResponse)
async def certificate_checkout(
    model_id: str,
    pack_name: str,
    db: AsyncSession = Depends(get_db),
    x_org_id: Optional[str] = Header(None)
):
    org = (await db.execute(select(Organization).filter(Organization.id == x_org_id))).scalars().first()
    if not org: raise HTTPException(status_code=404, detail="Org not found")

    if not org.stripe_customer_id:
        customer_id = StripeClient.create_customer(str(org.id), f"admin@{org.slug}.com", org.name)
        org.stripe_customer_id = customer_id
        await db.commit()

    checkout_url = StripeClient.create_compliance_checkout(
        customer_id=org.stripe_customer_id,
        model_id=model_id,
        pack_name=pack_name,
        success_url=f"{settings.ALLOWED_ORIGINS[0]}/dashboard/compliance?success=true",
        cancel_url=f"{settings.ALLOWED_ORIGINS[0]}/dashboard/compliance?canceled=true"
    )

    return CheckoutResponse(checkout_url=checkout_url)

@router.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature")
    
    try:
        event = StripeClient.construct_event(payload, sig_header)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if event["type"] == "invoice.payment_succeeded":
        # Activate subscription
        sub_id = event["data"]["object"]["subscription"]
        async with SessionLocal() as db:
            org = (await db.execute(select(Organization).filter(Organization.stripe_subscription_id == sub_id))).scalars().first()
            if org:
                org.subscription_status = "active"
                await db.commit()
                logger.info("Subscription activated", org_id=org.id)

    elif event["type"] == "invoice.payment_failed":
        # Handle failure (grace period)
        sub_id = event["data"]["object"]["subscription"]
        async with SessionLocal() as db:
            org = (await db.execute(select(Organization).filter(Organization.stripe_subscription_id == sub_id))).scalars().first()
            if org:
                org.subscription_status = "past_due"
                await db.commit()
                logger.warning("Subscription payment failed", org_id=org.id)

    return {"status": "success"}

@router.get("/subscription", response_model=SubscriptionDetails)
async def get_subscription(
    db: AsyncSession = Depends(get_db),
    x_org_id: Optional[str] = Header(None)
):
    org = (await db.execute(select(Organization).filter(Organization.id == x_org_id))).scalars().first()
    if not org: raise HTTPException(status_code=404, detail="Org not found")
    
    plan = (await db.execute(select(SubscriptionPlan).filter(SubscriptionPlan.id == org.plan_id))).scalars().first()
    plan_name = plan.name if plan else "Free"
    
    return SubscriptionDetails(
        plan_name=plan_name,
        status=org.subscription_status or "active",
        cancel_at_period_end=False
    )
