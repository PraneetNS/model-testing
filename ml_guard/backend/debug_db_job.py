
from app.db.session import SessionLocal
from app.db.models import Job
import sys

job_id = sys.argv[1]
db = SessionLocal()
job = db.get(Job, job_id)
if job:
    print(f"Job ID: {job.id}")
    print(f"Status: {job.status}")
    print(f"Model ID: {job.model_id}")
    print(f"Error: {job.error}")
else:
    print("Job not found in DB.")
db.close()
