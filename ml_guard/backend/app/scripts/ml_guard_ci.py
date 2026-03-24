import os
import sys
import time
import json
import requests
import argparse

"""
ML Guard CI/CD CLI.
Usage: 
  python ml_guard_cli.py --api-url http://ml-guard.io --api-key mlg_... --model-path model.pkl --data-path val.csv --model-name MyModel
"""

def run_ci_scan():
    parser = argparse.ArgumentParser(description="ML Guard CI/CD Pipeline Gate")
    parser.add_argument("--api-url", required=True, help="ML Guard API Base URL")
    parser.add_argument("--api-key", required=True, help="ML Guard API Key")
    parser.add_argument("--model-name", required=True, help="Name of the model being scanned")
    parser.add_argument("--model-path", required=True, help="Path to .pkl or .joblib model")
    parser.add_argument("--data-path", required=True, help="Path to validation .csv file")
    parser.add_argument("--label-col", default="target", help="Label column name in CSV")
    parser.add_argument("--poll-interval", type=int, default=5, help="Seconds between polls")
    parser.add_argument("--timeout", type=int, default=300, help="Max wait time in seconds")
    
    args = parser.parse_args()
    
    headers = {"X-Api-Key": args.api_key}
    
    print(f"🚀 Initializing ML Guard Governance Scan for '{args.model_name}'...")
    
    # 1. Trigger Scan
    try:
        with open(args.model_path, 'rb') as f_model, open(args.data_path, 'rb') as f_data:
            files = {
                "model_file": (os.path.basename(args.model_path), f_model),
                "val_file": (os.path.basename(args.data_path), f_data)
            }
            data = {
                "model_name": args.model_name,
                "label_col": args.label_col,
                "triggered_by": "ci"
            }
            response = requests.post(f"{args.api_url}/api/v1/audit/run", headers=headers, files=files, data=data)
            
        if response.status_code != 200:
            print(f"❌ Failed to start scan: {response.text}")
            sys.exit(1)
            
        job_id = response.json().get("job_id")
        print(f"✅ Scan started. Job ID: {job_id}")
        
    except Exception as e:
        print(f"❌ Error during initial request: {e}")
        sys.exit(1)
        
    # 2. Polling for results
    start_time = time.time()
    while time.time() - start_time < args.timeout:
        try:
            res = requests.get(f"{args.api_url}/api/v1/jobs/{job_id}", headers=headers)
            status_data = res.json()
            status = status_data.get("status")
            
            if status == "COMPLETED":
                gate = status_data.get("gate_status")
                score = status_data.get("governance_score")
                
                print("\n" + "="*40)
                print(f"📊 GOVERNANCE REPORT COMPLETE")
                print(f"Score: {score}/100")
                print(f"Gate Status: {gate}")
                print("="*40)
                
                if gate == "PASSED":
                    print("✅ Governance policy satisfied. Pipeline PASSED.")
                    sys.exit(0)
                elif gate == "WARNING":
                    print("⚠️ Governance policy has warnings. Proceed with caution.")
                    sys.exit(0) # Or 1 depending on strictness
                else:
                    print("❌ Governance policy REJECTED. Pipeline FAILED.")
                    sys.exit(1)
            
            elif status == "FAILED":
                print(f"❌ Job failed: {status_data.get('error')}")
                sys.exit(1)
                
            else:
                print(f"⏳ Scan in progress... ({status})")
                time.sleep(args.poll_interval)
                
        except Exception as e:
            print(f"⚠️ Polling error: {e}")
            time.sleep(args.poll_interval)
            
    print("❌ Scan timed out.")
    sys.exit(1)

if __name__ == "__main__":
    run_ci_scan()
