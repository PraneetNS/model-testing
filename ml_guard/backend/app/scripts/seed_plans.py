import asyncio
from app.db.session import SessionLocal
from app.db.models import SubscriptionPlan
from sqlalchemy.future import select
import structlog

logger = structlog.get_logger()

PLANS = [
    {
        "name": "Free",
        "slug": "free",
        "price_monthly_usd": 0,
        "predictions_per_month": 1000,
        "models_limit": 2,
        "reports_per_month": 1,
        "compliance_packs_limit": 0,
        "guardrail_eval_limit": 0,
        "is_custom_price": False
    },
    {
        "name": "Pro",
        "slug": "pro",
        "price_monthly_usd": 299,
        "predictions_per_month": 100000,
        "models_limit": -1, # unlimited
        "reports_per_month": -1, # unlimited
        "compliance_packs_limit": 2,
        "guardrail_eval_limit": 10000,
        "is_custom_price": False
    },
    {
        "name": "Enterprise",
        "slug": "enterprise",
        "price_monthly_usd": 0, # Custom
        "predictions_per_month": -1,
        "models_limit": -1,
        "reports_per_month": -1,
        "compliance_packs_limit": -1,
        "guardrail_eval_limit": -1,
        "is_custom_price": True
    }
]

async def seed_plans():
    from app.db.session import engine, Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    async with SessionLocal() as db:
        for p in PLANS:
            existing = (await db.execute(select(SubscriptionPlan).filter(SubscriptionPlan.slug == p["slug"]))).scalars().first()
            if not existing:
                plan = SubscriptionPlan(**p)
                db.add(plan)
                logger.info("Plan seeded", name=p["name"])
            else:
                # Update existing plan if needed
                for k, v in p.items():
                    setattr(existing, k, v)
                logger.info("Plan updated", name=p["name"])
        await db.commit()

if __name__ == "__main__":
    asyncio.run(seed_plans())
