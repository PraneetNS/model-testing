
import os
from dotenv import load_dotenv
from celery import Celery

env_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path=env_path, override=True)

# Manually load settings if needed, or just use the URL from .env
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    # Try CELERY_BROKER_URL
    REDIS_URL = os.getenv("CELERY_BROKER_URL")

print(f"Connecting to Broker: {REDIS_URL.split('@')[-1] if REDIS_URL else 'None'}")

app = Celery("ml_guard", broker=REDIS_URL)

try:
    insp = app.control.inspect()
    active = insp.active()
    
    if active:
        print("✅ Workers are ACTIVE:")
        for worker, tasks in active.items():
            print(f" - {worker}: {len(tasks)} tasks running")
    else:
        print("⚠️ No active workers found on this broker.")
        
    ping = insp.ping()
    if ping:
        print("✅ Workers responded to PING:")
        for worker, response in ping.items():
            print(f" - {worker}: {response}")
    else:
        print("❌ No workers responded to PING.")

except Exception as e:
    print(f"❌ Celery Inspect Failed: {e}")
