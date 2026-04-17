import asyncio
import hashlib
from app.db.session import SessionLocal
from app.db.models import APIKey, Organization
from sqlalchemy import select

async def ensure_key():
    db = SessionLocal()
    try:
        # 1. Ensure an organization exists
        org_res = await db.execute(select(Organization).limit(1))
        org = org_res.scalars().first()
        if not org:
            org = Organization(name="Default Org", slug="default-org")
            db.add(org)
            await db.commit()
            await db.refresh(org)
        
        target_key = "mlg_PeNfpwQSOtJkWr1Tow62Kr5luLuEugGi"
        key_hash = hashlib.sha256(target_key.encode()).hexdigest()
        
        # Check if exists
        res = await db.execute(select(APIKey).filter(APIKey.key_hash == key_hash))
        existing = res.scalars().first()
        
        if not existing:
            print(f"Creating missing API key for Next.js: {key_hash}")
            new_key = APIKey(
                org_id=org.id,
                label="Frontend Key",
                key_hash=key_hash,
                is_active=True,
                scopes=["admin", "ml_engineer", "auditor", "viewer"]
            )
            db.add(new_key)
            await db.commit()
            print("[INFO] API Key created successfully.")
        else:
            print("[INFO] API Key already exists and is valid.")
            if not existing.is_active:
                existing.is_active = True
                await db.commit()
                print("[INFO] API Key reactivated.")
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(ensure_key())
