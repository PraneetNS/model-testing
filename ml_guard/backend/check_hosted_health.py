
import requests
import json
import os
from dotenv import load_dotenv
import redis

load_dotenv()

API_URL = "https://ml-guard.onrender.com"
REDIS_URL = os.getenv("REDIS_URL")

def check_health():
    print(f"--- Checking Hosted API Health: {API_URL} ---")
    try:
        # 1. Base Health
        res = requests.get(f"{API_URL}/health")
        print(f"API Base: {res.status_code} - {res.text}")
        
        # 2. Database Health
        res = requests.get(f"{API_URL}/health/database")
        print(f"Database: {res.status_code} - {res.text}")
        
        # 3. Worker Health
        res = requests.get(f"{API_URL}/health/worker")
        print(f"Worker: {res.status_code} - {res.text}")
        
    except Exception as e:
        print(f"❌ API Health Check Failed: {e}")

def check_redis():
    print(f"\n--- Checking Redis (Upstash): {REDIS_URL.split('@')[-1] if REDIS_URL else 'None'} ---")
    if not REDIS_URL:
        print("❌ REDIS_URL not found in .env")
        return
        
    try:
        r = redis.from_url(REDIS_URL)
        ping = r.ping()
        print(f"Redis Ping: {'✅ Success' if ping else '❌ Failed'}")
        
        # Check for active workers (Celery stores worker info in Redis)
        # Note: This is an approximation since we don't have celery inspect access easily without celery installed
        keys = r.keys("*")
        print(f"Total keys in Redis: {len(keys)}")
        
        # Celery usually has keys like 'celery-task-meta-*' or worker heartbeats
        worker_keys = [k.decode('utf-8') for k in keys if "worker" in k.decode('utf-8') or "celery" in k.decode('utf-8')]
        print(f"Worker related keys: {worker_keys[:10]}")
        
    except Exception as e:
        print(f"❌ Redis Connection Failed: {e}")

if __name__ == "__main__":
    check_health()
    check_redis()
