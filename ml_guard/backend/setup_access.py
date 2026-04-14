import asyncio
import os
import sys
import secrets
import string
import uuid

# Add backend to path
sys.path.append(os.getcwd())

from app.db.session import AsyncSessionLocal
from app.db.models import APIKey, Organization, User
from passlib.context import CryptContext
from sqlalchemy import select

ctx = CryptContext(schemes=["pbkdf2_sha256"])

async def setup_default_access():
    async with AsyncSessionLocal() as db:
        # 1. Create Default Org
        org_stmt = select(Organization).limit(1)
        org = (await db.execute(org_stmt)).scalars().first()
        if not org:
            org = Organization(
                name="ML Guard Enterprise",
                slug="ml-guard-enterprise"
            )
            db.add(org)
            await db.flush()
        
        # 2. Create Default API Key
        alphabet = string.ascii_letters + string.digits
        raw_key = "mlg_" + "".join(secrets.choice(alphabet) for _ in range(32))
        key_hash = ctx.hash(raw_key)
        
        new_key = APIKey(
            org_id=org.id,
            label="Default Admin Key",
            key_hash=key_hash,
            is_active=True,
            scopes=["audit", "behavior", "monitor", "admin"]
        )
        db.add(new_key)
        await db.commit()
        
        return raw_key

if __name__ == "__main__":
    key = asyncio.run(setup_default_access())
    print(f"NEW_VALID_API_KEY: {key}")
