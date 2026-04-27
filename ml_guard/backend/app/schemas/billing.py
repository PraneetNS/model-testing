from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from uuid import UUID
from datetime import datetime

class UsageStat(BaseModel):
    event_type: str
    used: int
    limit: int
    pct_used: float
    overage_units: int
    overage_cost_usd: float

class BillingUsageResponse(BaseModel):
    organization_id: UUID
    plan_name: str
    usage: List[UsageStat]

class SubscriptionDetails(BaseModel):
    plan_name: str
    status: str
    current_period_end: Optional[datetime] = None
    cancel_at_period_end: bool = False

class CheckoutResponse(BaseModel):
    checkout_url: str

class SubscribeResponse(BaseModel):
    subscription_id: str
    client_secret: str
