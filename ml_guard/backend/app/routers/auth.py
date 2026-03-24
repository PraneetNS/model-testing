from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import hashlib
from app.db.session import get_db
from app.db.models import APIKey, Organization, User, generate_api_key, utcnow
from app.core.auth import require_admin, AuthContext

router = APIRouter()

@router.post("/auth/apikey")
async def create_api_key(
    label: str = "CI/CD Key",
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_admin)
):
    """Generate a new API key for the organization."""
    raw_key = generate_api_key()
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    
    new_key = APIKey(
        org_id=auth.org_id,
        label=label,
        key_hash=key_hash,
        is_active=True,
        scopes=["audit", "behavior", "monitor"]
    )
    db.add(new_key)
    db.commit()
    
    return {
        "label": label,
        "api_key": raw_key,
        "note": "Store this safely. It will not be shown again."
    }

@router.get("/auth/apikeys")
async def list_api_keys(
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_admin)
):
    """List active API keys (hashes only)."""
    keys = db.query(APIKey).filter(APIKey.org_id == auth.org_id).all()
    return [
        {
            "id": str(k.id),
            "label": k.label,
            "is_active": k.is_active,
            "last_used": str(k.last_used) if k.last_used else None,
            "created_at": str(k.created_at)
        } for k in keys
    ]
