import asyncio
import os
import sys
import secrets
import string

# Add backend to path
sys.path.append(os.getcwd())

from app.db.session import AsyncSessionLocal
from app.db.models import APIKey
from app.core.security import get_password_hash

async def create_default_key():
    alphabet = string.ascii_letters + string.digits
    raw_key = "mlg_" + "".join(secrets.choice(alphabet) for _ in range(32))
    
    # Use pbkdf2_sha256 if bcrypt fails
    from passlib.context import CryptContext
    ctx = CryptContext(schemes=["pbkdf2_sha256"])
    key_hash = ctx.hash(raw_key)
    
    async with AsyncSessionLocal() as db:
        new_key = APIKey(
            label="Default Admin Key",
            key_hash=key_hash,
            is_active=True,
            owner_id=None,
            role="admin"
        )
        db.add(new_key)
        await db.commit()
    
    print(f"NEW_KEY_CREATED: {raw_key}")

if __name__ == "__main__":
    asyncio.run(create_default_key())
