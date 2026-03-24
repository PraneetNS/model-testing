
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.core.celery_app import celery_app

i = celery_app.control.inspect()
tasks = i.registered_tasks()
print(f"Registered tasks: {tasks}")
