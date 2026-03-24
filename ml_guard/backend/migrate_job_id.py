
from app.db.session import engine
from sqlalchemy import text

with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE scan_records ADD COLUMN job_id VARCHAR(50)"))
        conn.commit()
        print("Successfully added job_id column to scan_records.")
    except Exception as e:
        print(f"Error or already exists: {e}")
