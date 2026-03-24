
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.core.celery_app import test_task
import time

print("Dispatching test task...")
result = test_task.delay()
print(f"Task ID: {result.id}")

for _ in range(10):
    status = result.status
    print(f"Status: {status}")
    if status == 'SUCCESS':
        print(f"Result: {result.result}")
        break
    time.sleep(1)
else:
    print("Task timed out or is stuck.")
