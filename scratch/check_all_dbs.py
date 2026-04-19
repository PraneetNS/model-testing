import sqlite3

db_paths = [
    r"c:\Users\savan\OneDrive\Desktop\real_Fireflink_ML\ml_guard\backend\ml_guard.db",
    r"c:\Users\savan\OneDrive\Desktop\real_Fireflink_ML\ml_guard\ml_guard.db",
    r"c:\Users\savan\OneDrive\Desktop\real_Fireflink_ML\ml_guard.db"
]

for db_path in db_paths:
    print(f"Checking {db_path}...")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(models)")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"  Columns: {columns}")
        conn.close()
    except Exception as e:
        print(f"  Error: {e}")
