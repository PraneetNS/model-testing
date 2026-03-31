import sys
import os
import hashlib
import uuid
from sqlalchemy import text

# Core path setup
sys.path.insert(0, os.path.dirname(__file__))

from app.db.session import engine, Base, SessionLocal
from app.db.models import generate_api_key

def seed():
    print("Initializing Multi-Tenant Governance Platform Schema...")
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Check if already seeded
        res = db.execute(text("SELECT id FROM organizations WHERE slug = 'mlguard-dev'")).fetchone()
        if res:
            print("(!) Found existing organization: mlguard-dev. Re-fetching IDs...")
            org_id = res[0]
            model_res = db.execute(text(f"SELECT id FROM models WHERE name = 'TestChurnModel-v1'")).fetchone()
            model_id = model_res[0] if model_res else None
            
            # Write IDs anyway
            env_dev_path = os.path.join(os.path.dirname(__file__), ".env.dev")
            # Note: Cannot fetch raw key hash from here conveniently
            return

        print("Executing cold-seed via SQL Core...")
        
        org_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        project_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        key_id = str(uuid.uuid4())
        
        # 1. Organization
        db.execute(text(
            "INSERT INTO organizations (id, name, slug, plan, settings, created_at) "
            "VALUES (:id, :name, :slug, :plan, :settings, datetime('now'))"
        ), {"id": org_id, "name": "ML Guard Dev", "slug": "mlguard-dev", "plan": "enterprise", "settings": "{}"})
        
        # 2. User
        db.execute(text(
            "INSERT INTO users (id, org_id, email, name, role, auth_provider, password_hash, is_active, created_at) "
            "VALUES (:id, :org_id, :email, :name, :role, :auth_provider, :password_hash, :is_active, datetime('now'))"
        ), {"id": user_id, "org_id": org_id, "email": "admin@mlguard.local", "name": "ML Guard Admin", "role": "admin", "auth_provider": "local", "password_hash": "seeded", "is_active": True})
        
        # 3. Project
        db.execute(text(
            "INSERT INTO projects (id, org_id, name, description, created_by, created_at) "
            "VALUES (:id, :org_id, :name, :description, :created_by, datetime('now'))"
        ), {"id": project_id, "org_id": org_id, "name": "E2E Rescue Project", "description": "validation", "created_by": user_id})
        
        # 4. API Key
        raw_key = generate_api_key()
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        db.execute(text(
            "INSERT INTO api_keys (id, org_id, key_hash, label, scopes, is_active, created_at) "
            "VALUES (:id, :org_id, :key_hash, :label, :scopes, :is_active, datetime('now'))"
        ), {"id": key_id, "org_id": org_id, "key_hash": key_hash, "label": "Dev Key", "scopes": '["audit", "behavior", "monitor", "governance", "admin"]', "is_active": True})
        
        # 5. Model
        db.execute(text(
            "INSERT INTO models (id, project_id, name, provider, metadata_json, version, created_at, created_by) "
            "VALUES (:id, :project_id, :name, :provider, :metadata_json, 1, datetime('now'), :created_by)"
        ), {"id": model_id, "project_id": project_id, "name": "TestChurnModel-v1", "provider": "sklearn", "metadata_json": "{}", "created_by": user_id})
        
        db.commit()
        print("OK Database seeded with Multi-Tenant core data.")

        # Save to .env.dev for localized test scripts
        env_dev_path = os.path.join(os.path.dirname(__file__), ".env.dev")
        with open(env_dev_path, "w") as f:
            f.write(f"DEV_API_KEY={raw_key}\n")
            f.write(f"DEV_MODEL_ID={model_id}\n")
            f.write(f"DEV_ORG_ID={org_id}\n")
            f.write(f"DEV_PROJECT_ID={project_id}\n")
        print(f"OK metadata saved to: {env_dev_path}")
        print(f"\nAPI KEY: {raw_key}")
        
    except Exception as e:
        import traceback
        print(f"(!) ERROR seed failed: {e}")
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed()
