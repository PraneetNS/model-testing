import sqlite3

def migrate():
    conn = sqlite3.connect('ml_guard.db')
    cursor = conn.cursor()
    
    # 1. Add model_id
    try:
        print("Adding model_id to alerts...")
        cursor.execute("ALTER TABLE alerts ADD COLUMN model_id CHAR(36)")
    except Exception as e:
        print(f"Skipped model_id: {e}")

    # 2. Add severity
    try:
        print("Adding severity to alerts...")
        cursor.execute("ALTER TABLE alerts ADD COLUMN severity VARCHAR(20)")
    except Exception as e:
        print(f"Skipped severity: {e}")

    # 3. Add created_at
    try:
        print("Adding created_at to alerts...")
        cursor.execute("ALTER TABLE alerts ADD COLUMN created_at DATETIME")
        # Initialize created_at with timestamp values
        cursor.execute("UPDATE alerts SET created_at = timestamp")
    except Exception as e:
        print(f"Skipped created_at: {e}")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
