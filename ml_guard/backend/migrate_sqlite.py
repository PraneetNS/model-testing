import sqlite3

def migrate():
    conn = sqlite3.connect('ml_guard.db')
    cursor = conn.cursor()
    
    columns_to_add = [
        ("risk_score", "INTEGER"),
        ("risk_level", "VARCHAR(20)"),
        ("top_drifted_features", "JSON"),
        ("fairness_metrics", "JSON"),
        ("bias_violation_flag", "BOOLEAN"),
        ("fairness_risk_score", "FLOAT")
    ]
    
    cursor.execute("PRAGMA table_info(scan_records)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    
    for col_name, col_type in columns_to_add:
        if col_name not in existing_cols:
            print(f"Adding column {col_name} to scan_records...")
            try:
                cursor.execute(f"ALTER TABLE scan_records ADD COLUMN {col_name} {col_type}")
            except Exception as e:
                print(f"Error adding {col_name}: {e}")
    
    # AuditLog might also need org_id, user_id, ip_address if they were updated
    cursor.execute("PRAGMA table_info(audit_logs)")
    existing_audit_cols = [row[1] for row in cursor.fetchall()]
    audit_cols = [
        ("org_id", "CHAR(36)"),
        ("user_id", "CHAR(36)"),
        ("ip_address", "VARCHAR(45)")
    ]
    for col_name, col_type in audit_cols:
         if col_name not in existing_audit_cols:
            print(f"Adding column {col_name} to audit_logs...")
            try:
                cursor.execute(f"ALTER TABLE audit_logs ADD COLUMN {col_name} {col_type}")
            except Exception as e:
                print(f"Error adding {col_name}: {e}")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
