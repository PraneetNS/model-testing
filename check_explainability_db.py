
import asyncio
import os
import sys

# Move to backend directory to ensure ./ml_guard.db is the correct one
backend_dir = os.path.join(os.getcwd(), "ml_guard", "backend")
os.chdir(backend_dir)
sys.path.append(backend_dir)

# Force load .env from the current (backend) location
from dotenv import load_dotenv
load_dotenv(".env")

async def check_db():
    from app.db.session import SessionLocal
    from app.db.models import ExplainabilityResult, Model
    from sqlalchemy.future import select
    from app.core.config import settings
    
    print(f"Working Dir: {os.getcwd()}")
    print(f"Using Database URL: {settings.DATABASE_URL}")
    
    db = SessionLocal()
    try:
        # Check Models
        res_models = await db.execute(select(Model))
        models = res_models.scalars().all()
        print(f"Total Models in DB: {len(models)}")
        for m in models:
            print(f" - Model: {m.name} (ID: {m.id})")

        # Check Results
        res = await db.execute(select(ExplainabilityResult))
        records = res.scalars().all()
        print(f"\nTotal Explainability Results: {len(records)}")
        for r in records:
            print(f" - ID: {r.id}, Model ID: {r.model_id}, Method: {r.method}, Created: {r.created_at}")
            
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(check_db())
