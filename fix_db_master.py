import sqlite3
import uuid
from datetime import datetime

def fix_all():
    conn = sqlite3.connect('ml_guard/backend/ml_guard.db')
    curr = conn.cursor()

    # 1. Fix prediction_logs schema
    try:
        curr.execute("ALTER TABLE prediction_logs ADD COLUMN model_id VARCHAR")
        print("Added model_id to prediction_logs")
    except sqlite3.OperationalError:
        print("model_id already exists in prediction_logs")

    # 2. Seed an Organization if not exists
    org_id = "6adf911b-a696-4acf-b366-c1d2008919bf"
    curr.execute("INSERT OR IGNORE INTO organizations (id, name, created_at) VALUES (?, ?, ?)", 
                 (org_id, "Fireflink Enterprise", datetime.now().isoformat()))

    # 3. Seed an Active Policy
    policy_id = str(uuid.uuid4())
    config = '{"min_governance_score": 70, "max_psi": 0.2, "min_accuracy": 0.75}'
    curr.execute("""
        INSERT INTO policy_versions (id, name, version, config, is_active, created_at, org_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (policy_id, "Standard Governance Policy", "1.0.0", config, 1, datetime.now().isoformat(), org_id))
    print(f"Seeded active policy: {policy_id}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    fix_all()
