import requests
import json
import os
import time

BASE = "http://127.0.0.1:8000"
API_KEY = "mlg_1Ai7zfmfsB_GLaoNuKjOOopFh12xLzGy7SDqh7Kho1U"
MODEL_ID = "f9597635-5c66-4b17-9e4b-38e3fde81a53"
MODEL_NAME = "TestChurnModel-v1"

# Adjust paths to your actual file locations
MODEL_FILE = r"C:\Users\savan\OneDrive\Desktop\real_Fireflink_ML\test_model.pkl"
TRAIN_FILE = r"C:\Users\savan\OneDrive\Desktop\real_Fireflink_ML\test_train.csv"
VAL_FILE   = r"C:\Users\savan\OneDrive\Desktop\real_Fireflink_ML\test_val.csv"

HEADERS = {"X-API-Key": API_KEY}

def get_governance_score():
    r = requests.get(
        f"{BASE}/api/v1/governance/{MODEL_ID}/score",
        headers=HEADERS,
        timeout=30
    )
    if r.status_code == 200:
        d = r.json()
        return d.get("overall_score"), d.get("live_score"), d.get("verdict")
    return None, None, None

def run_full_audit():
    print("\n── Running Full Audit Scan ────────")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Checks: drift, performance, fairness, security")
    
    with open(MODEL_FILE, "rb") as mf, \
         open(TRAIN_FILE, "rb") as tf, \
         open(VAL_FILE, "rb") as vf:
        
        files = {
            "model_file": ("test_model.pkl", mf, 
                          "application/octet-stream"),
            "train_file": ("test_train.csv", tf, 
                          "text/csv"),
            "val_file":   ("test_val.csv", vf, 
                          "text/csv"),
        }
        data = {
            "model_name":  MODEL_NAME,
            "label_col":   "target",
            "selected":    ["drift", "performance", 
                           "fairness", "security"],
        }
        
        print("  Sending audit request...")
        start = time.time()
        
        r = requests.post(
            f"{BASE}/api/v1/audit/run",
            headers=HEADERS,
            files=files,
            data=data,
            timeout=120  # audits can take time
        )
        
        elapsed = time.time() - start
        print(f"  Response: {r.status_code} "
              f"({elapsed:.1f}s)")
        
        if r.status_code == 200:
            return r.json()
        else:
            print(f"  Error: {r.text[:500]}")
            return None

def run_drift_scan():
    """Log predictions then check drift."""
    print("\n── Logging Sample Predictions ─────")
    import csv
    
    with open(VAL_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)[:20]  # first 20 rows
    
    logged = 0
    for row in rows:
        payload = {
            "model_id": MODEL_ID,
            "features": {
                "f1": float(row["f1"]),
                "f2": float(row["f2"]),
                "f3": float(row["f3"]),
            },
            "prediction": row["target"],
            "prediction_proba": 0.75,
            "latency_ms": 12.5,
            "environment": "production",
            "ground_truth": row["target"]
        }
        r = requests.post(
            f"{BASE}/api/v1/ingest/predict",
            headers={**HEADERS, 
                     "Content-Type": "application/json"},
            json=payload,
            timeout=10
        )
        if r.status_code in [200, 202]:
            logged += 1
    
    print(f"✓ Logged {logged}/20 predictions")

def generate_certificate():
    print("\n── Generating Certificate ──────────")
    r = requests.post(
        f"{BASE}/api/v1/governance/{MODEL_ID}/certify",
        headers={**HEADERS, 
                 "Content-Type": "application/json"},
        json={"force_regenerate": True},
        timeout=30
    )
    if r.status_code == 200:
        d = r.json()
        print(f"✓ cert_hash: {d.get('cert_hash','')[:32]}...")
        print(f"✓ Verdict:   {d.get('verdict')}")
        print(f"✓ Score:     {d.get('overall_score')}")
        return d.get("cert_hash")
    else:
        print(f"  Error: {r.text[:200]}")
        return None

if __name__ == "__main__":
    print("═══════════════════════════════════════")
    print("  ML Guard v7.2 — Real Audit Runner    ")
    print("═══════════════════════════════════════")
    
    # Step 1: Get baseline score BEFORE audit
    print("\n── Baseline Governance Score ───────")
    score, live, verdict = get_governance_score()
    print(f"  Before audit:")
    print(f"  Score:   {score}")
    print(f"  Live:    {live}")
    print(f"  Verdict: {verdict}")
    
    # Step 2: Log real predictions for drift
    run_drift_scan()
    
    # Step 3: Run full audit scan
    result = run_full_audit()
    
    if result:
        print("\n── Audit Results ───────────────────")
        # Print key metrics from audit
        if isinstance(result, dict):
            gov = result.get("governance", {})
            perf = result.get("performance", {})
            drift = result.get("drift", {})
            fairness = result.get("fairness", {})
            
            if gov:
                print(f"✓ Governance Score: "
                      f"{gov.get('score', 'N/A')}")
            if perf:
                acc = perf.get("accuracy", 
                    perf.get("accuracy_score", "N/A"))
                print(f"✓ Accuracy: {acc}")
            if drift:
                psi = drift.get("psi", 
                    drift.get("overall_psi", "N/A"))
                print(f"✓ PSI Drift: {psi}")
            
            print(f"\n  Full result keys: "
                  f"{list(result.keys())}")
    
    # Step 4: Get score AFTER audit
    print("\n── Post-Audit Governance Score ─────")
    time.sleep(2)  # brief wait for DB writes
    score2, live2, verdict2 = get_governance_score()
    print(f"  After audit:")
    print(f"  Score:   {score2}")
    print(f"  Live:    {live2}")
    print(f"  Verdict: {verdict2}")
    
    if score and score2:
        delta = score2 - score
        print(f"\n  Score delta: "
              f"{'+' if delta >= 0 else ''}{delta:.2f}")
    
    # Step 5: Generate fresh certificate
    cert_hash = generate_certificate()
    
    if cert_hash:
        print(f"\n  Verify URL:")
        print(f"  http://localhost:3000/verify/{cert_hash}")
    
    print("\n═══════════════════════════════════════")
    print("  ✓ Audit Run Complete                 ")
    print("═══════════════════════════════════════")
