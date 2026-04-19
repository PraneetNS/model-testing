
import asyncio
import os
import sys
import hashlib

# Setup environment
backend_dir = os.path.join(os.getcwd(), "ml_guard", "backend")
if backend_dir not in sys.path:
    sys.path.append(backend_dir)
os.chdir(backend_dir)

from dotenv import load_dotenv
load_dotenv(".env")

async def setup_dev_key():
    from app.db.session import AsyncSessionLocal
    from app.db.models import APIKey, Organization
    from sqlalchemy.future import select
    from app.core.config import settings
    
    db = AsyncSessionLocal()
    try:
        # 1. Ensure an organization exists
        res_org = await db.execute(select(Organization).limit(1))
        org = res_org.scalars().first()
        if not org:
            org = Organization(name="Default Org", slug="default-org")
            db.add(org)
            await db.flush()
            print(f"Created Default Org: {org.id}")
        
        # 2. Check for the specific simulator key
        SIMULATOR_KEY = "mlg_simulator_key_2026_safe_dev"
        key_hash = hashlib.sha256(SIMULATOR_KEY.encode()).hexdigest()
        
        res_key = await db.execute(select(APIKey).filter(APIKey.key_hash == key_hash))
        existing_key = res_key.scalars().first()
        
        if not existing_key:
            new_key = APIKey(
                org_id=org.id,
                key_hash=key_hash,
                label="Simulator Key (Dev)",
                is_active=True,
                scopes=["audit", "behavior", "monitor", "ingest", "security"]
            )
            db.add(new_key)
            await db.commit()
            print(f"Created NEW active simulator key: {SIMULATOR_KEY}")
        else:
            if not existing_key.is_active:
                existing_key.is_active = True
                await db.commit()
                print("Re-activated simulator key.")
            else:
                print("Simulator key is already active.")
                
        return SIMULATOR_KEY
            
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(setup_dev_key())
