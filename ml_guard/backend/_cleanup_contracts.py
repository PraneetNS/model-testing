import sqlite3
MODEL_ID = "f9597635-5c66-4b17-9e4b-38e3fde81a53"
conn = sqlite3.connect("ml_guard.db")
cur = conn.cursor()

# Count before
cur.execute("SELECT COUNT(*) FROM model_contracts")
print("contracts before:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM contract_breaches")
print("breaches before:", cur.fetchone()[0])

# Find the LATEST contract to keep
cur.execute(
    "SELECT id FROM model_contracts WHERE model_id=? ORDER BY created_at DESC LIMIT 1",
    (MODEL_ID,)
)
row = cur.fetchone()
if row:
    keep_id = row[0]
    print("Keeping contract:", keep_id)
    # Delete all others
    cur.execute(
        "DELETE FROM model_contracts WHERE model_id=? AND id != ?",
        (MODEL_ID, keep_id)
    )
    print("deleted duplicate contracts:", cur.rowcount)

# Clear all breach records for a clean test
cur.execute("DELETE FROM contract_breaches")
print("cleared breach records:", cur.rowcount)

conn.commit()

# Count after
cur.execute("SELECT COUNT(*) FROM model_contracts")
print("contracts after:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM contract_breaches")
print("breaches after:", cur.fetchone()[0])
conn.close()
print("Cleanup complete")
