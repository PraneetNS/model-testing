
import sqlite3
import json

def check_db():
    conn = sqlite3.connect('ml_guard.db')
    curr = conn.cursor()
    
    print("--- Scan Records ---")
    curr.execute("SELECT id, scan_type, trigger_source, created_at, gate_status FROM scan_records ORDER BY created_at DESC LIMIT 5")
    for row in curr.fetchall():
        print(row)
    
    print("\n--- Latest Jobs ---")
    curr.execute("SELECT id, status, error, created_at FROM jobs ORDER BY created_at DESC LIMIT 5")
    for row in curr.fetchall():
        print(row)
        
    print("\n--- Projects ---")
    curr.execute("SELECT id, name, org_id FROM projects")
    for row in curr.fetchall():
        print(row)
        
    conn.close()

if __name__ == "__main__":
    check_db()
