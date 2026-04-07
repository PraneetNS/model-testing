import sqlite3
import sys

conn = sqlite3.connect("ml_guard.db")
cur = conn.cursor()

# Get existing columns
cur.execute("PRAGMA table_info(prediction_logs)")
existing = {row[1] for row in cur.fetchall()}

# Columns to add if missing
missing = {
    "model_id": "VARCHAR(255)",
    "model_version_id": "CHAR(36)",
    "timestamp": "DATETIME",
    "prediction_proba": "FLOAT",
    "ground_truth": "VARCHAR(255)",
    "latency_ms": "FLOAT",
    "data_source": "VARCHAR(50)",
    "environment": "VARCHAR(50)",
    "tags": "JSON"
}

for col, dtype in missing.items():
    if col not in existing:
        print(f"Adding column {col}...")
        try:
            cur.execute(f"ALTER TABLE prediction_logs ADD COLUMN {col} {dtype}")
        except Exception as e:
            print(f"Error adding {col}: {e}")

conn.commit()
conn.close()
print("Migration check complete.")
