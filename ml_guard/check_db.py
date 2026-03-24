
import os
import sys
from sqlalchemy import create_engine, inspect

# Add backend to path
base_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(base_dir, "backend")
if os.path.exists(backend_dir):
    sys.path.append(backend_dir)

from app.core.config import settings

def check_tables():
    uri = settings.SQLALCHEMY_DATABASE_URI
    print(f"Connecting to: {uri}")
    engine = create_engine(uri)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tables found: {tables}")

if __name__ == "__main__":
    check_tables()
