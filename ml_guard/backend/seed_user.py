import sys
from pathlib import Path

# Add the backend directory to sys.path to allow imports
sys.path.append(str(Path(__file__).parent))

from app.infrastructure.database import SessionLocal
from app.infrastructure.persistence import models as sql_models
from app.core import security
from app.core.config import settings

def seed_default_user():
    print(f"Seeding with DB URI: {settings.SQLALCHEMY_DATABASE_URI}")
    db = SessionLocal()
    try:
        # Check if tenant exists
        tenant_name = "Fireflink Enterprise"
        tenant = db.query(sql_models.Tenant).filter(sql_models.Tenant.name == tenant_name).first()
        if not tenant:
            tenant = sql_models.Tenant(name=tenant_name)
            db.add(tenant)
            db.commit()
            db.refresh(tenant)
            print(f"Tenant '{tenant_name}' created.")
        else:
            print(f"Tenant '{tenant_name}' already exists.")

        # Check if user exists
        email = "admin@fireflink.com"
        password = "password123"
        user = db.query(sql_models.User).filter(sql_models.User.email == email).first()
        if not user:
            user = sql_models.User(
                email=email,
                hashed_password=security.get_password_hash(password),
                full_name="System Administrator",
                role="admin",
                tenant_id=tenant.id,
                is_active=True
            )
            db.add(user)
            db.commit()
            print(f"User '{email}' created with password '{password}'.")
        else:
            print(f"User '{email}' already exists.")
            # Update password just in case
            user.hashed_password = security.get_password_hash(password)
            db.commit()
            print(f"User '{email}' password updated to '{password}'.")

    except Exception as e:
        import traceback
        print(f"Error seeding user: {e}")
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_default_user()
