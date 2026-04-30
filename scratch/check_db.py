import sqlite3
import os

db_path = "ml_guard/backend/ml_guard.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(prediction_logs)")
    columns = cursor.fetchall()
    print("Columns in prediction_logs:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    conn.close()
else:
    print(f"File not found: {db_path}")
