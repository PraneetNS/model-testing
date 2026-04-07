import sqlite3
conn = sqlite3.connect("ml_guard.db")
cur = conn.cursor()
cur.execute("PRAGMA table_info(prediction_logs)")
cols = cur.fetchall()
for c in cols:
    print(c)
conn.close()
