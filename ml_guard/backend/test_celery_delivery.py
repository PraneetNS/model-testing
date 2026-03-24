
import os
from dotenv import load_dotenv
from app.core.celery_app import test_task
import time

load_dotenv(override=True)

print("Sending test task...")
result = test_task.delay()
print(f"Task ID: {result.id}")

for _ in range(10):
    status = result.status
    print(f"Status: {status}")
    if result.ready():
        print(f"Result: {result.get()}")
        break
    time.sleep(2)
else:
    print("Task timed out. No worker picked it up.")
