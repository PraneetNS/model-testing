from app.db.session import SessionLocal
from app.db.models import ScanRecord

db = SessionLocal()
scan = db.query(ScanRecord).order_by(ScanRecord.created_at.desc()).first()
if scan:
    print(f"LATEST_SCAN_ID: {scan.id}")
else:
    print("NO_SCANS")
db.close()
