import sys
sys.path.insert(0, ".")
from app.db.session import engine, Base
from app.db.models import ModelContract, ContractBreach

Base.metadata.create_all(bind=engine)
print("Tables created OK")

import sqlite3
conn = sqlite3.connect("ml_guard.db")
cur = conn.cursor()
cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' "
    "AND name IN ('model_contracts','contract_breaches')"
)
rows = cur.fetchall()
print("Verified in DB:", [r[0] for r in rows])
conn.close()
