from app.db.session import SessionLocal
from app.db.models import ScanRecord
import json

db = SessionLocal()
try:
    scans = db.query(ScanRecord).filter(ScanRecord.security_checks.isnot(None)).all()
    for s in scans:
        print(f"ID: {s.id}")
        print(f"Type of security_checks: {type(s.security_checks)}")
        print(f"Value: {s.security_checks}")
finally:
    db.close()
