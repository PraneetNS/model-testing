import sqlite3

db_path = r"c:\Users\savan\OneDrive\Desktop\real_Fireflink_ML\ml_guard\backend\ml_guard.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("PRAGMA table_info(models)")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Columns in 'models' table: {columns}")
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()
