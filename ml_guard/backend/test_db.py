import sqlalchemy
from sqlalchemy import create_engine, text
import os

db_url = "postgresql+psycopg2://neondb_owner:npg_SIM4XQpEFdb1@ep-delicate-mud-aizxi3by-pooler.c-4.us-east-1.aws.neon.tech/neondb?sslmode=require"

engine = create_engine(db_url)
try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print(f"DB Connection Successful: {result.fetchone()[0]}")
except Exception as e:
    print(f"DB Connection Failed: {e}")
