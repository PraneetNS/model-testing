import asyncio
import os
import sys
import secrets
import string
from passlib.context import CryptContext

# Add backend to path
sys.path.append(os.getcwd())

from app.db.session import AsyncSessionLocal
from app.db.models import Organization, User, APIKey

async def create_key_and_org():
    async with AsyncSessionLocal() as db:
        from sqlalchemy.future import select
        
        # 1. Ensure Organization exists
        res = await db.execute(select(Organization).filter(Organization.slug == "default"))
        org = res.scalars().first()
        if not org:
            org = Organization(name="Default Organization", slug="default")
            db.add(org)
            await db.commit()
            await db.refresh(org)
            print(f"Created Org: {org.id}")
        else:
            print(f"Found Org: {org.id}")

        # 2. Ensure Admin User exists
        res = await db.execute(select(User).filter(User.email == "admin@mlguard.io"))
        user = res.scalars().first()
        if not user:
            user = User(
                org_id=org.id,
                email="admin@mlguard.io",
                name="System Admin",
                role="admin",
                is_active=True
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            print(f"Created User: {user.id}")
        else:
            print(f"Found User: {user.id}")

        # 3. Create API Key
        alphabet = string.ascii_letters + string.digits
        raw_key = "mlg_" + "".join(secrets.choice(alphabet) for _ in range(32))
        
        ctx = CryptContext(schemes=["pbkdf2_sha256"])
        key_hash = ctx.hash(raw_key)
        
        new_key = APIKey(
            org_id=org.id,
            label="Admin Access Key",
            key_hash=key_hash,
            is_active=True,
            scopes=["admin", "ml_engineer", "auditor", "viewer"]
        )
        db.add(new_key)
        await db.commit()
        
        print(f"NEW_KEY_CREATED: {raw_key}")

if __name__ == "__main__":
    asyncio.run(create_key_and_org())
