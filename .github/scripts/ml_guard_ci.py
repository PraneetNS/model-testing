"""
ml_guard_ci.py — ML Guard CI/CD Governance Gate v7.2

Production-grade CI script that:
1. Checks API health
2. Runs governance audit scan (multipart upload)
3. Runs synchronous governance gate check
4. Checks contract breach summary (24h)
5. Writes governance_result.json for PR comments
6. Exits 0 (pass) or 1 (fail) for pipeline control

Usage:
  python ml_guard_ci.py \\
    --api-url https://your-mlguard.onrender.com \\
    --api-key mlg_xxx \\
    --model-name MyModel \\
    --model-path model.pkl \\
    --data-path val.csv \\
    --min-score 60 \\
    --strict false
"""

import os
import sys
import time
import json
import argparse
import requests
from typing import Optional


def parse_args():
    p = argparse.ArgumentParser(
        description="ML Guard CI/CD Governance Gate"
    )
    p.add_argument("--api-url", required=True,
        help="ML Guard API base URL")
    p.add_argument("--api-key", required=True,
        help="ML Guard API key (mlg_...)")
    p.add_argument("--model-name", required=True,
        help="Name of the model being scanned")
    p.add_argument("--model-path", required=True,
        help="Path to .pkl or .joblib model file")
    p.add_argument("--data-path", required=True,
        help="Path to validation .csv file")
    p.add_argument("--label-col", default="target",
        help="Label column name in CSV (default: target)")
    p.add_argument("--min-score", type=float, default=60.0,
        help="Minimum governance score to pass (default: 60)")
    p.add_argument("--strict", default="false",
        help="Fail on CONDITIONAL verdict too (true/false)")
    p.add_argument("--timeout", type=int, default=300,
        help="Request timeout in seconds (default: 300)")
    return p.parse_args()


# Use ASCII separator so it works on any terminal (Windows cp1252, etc.)
SEP = "=" * 52


def get_headers(api_key: str) -> dict:
    """
    Returns auth headers with the correct header name.
    FIX: X-API-Key (not X-Api-Key -- that was the bug).
    Content-Type is omitted here so multipart requests
    let requests set it automatically with boundary.
    """
    return {
        "X-API-Key": api_key,
    }


def get_json_headers(api_key: str) -> dict:
    """Headers for JSON body requests."""
    return {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }


def check_health(base_url: str, api_key: str) -> bool:
    """
    Ping /health to verify the API is reachable and healthy.
    Returns True if healthy, False otherwise.
    """
    try:
        r = requests.get(
            f"{base_url}/health",
            headers=get_headers(api_key),
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            v = data.get("version", "unknown")
            status = data.get("status", "ok")
            print(f"  [OK] ML Guard API healthy (v{v}, status={status})")
            return True
        print(f"  [FAIL] API unhealthy: HTTP {r.status_code}")
        return False
    except requests.ConnectionError:
        print(f"  [FAIL] Cannot connect to {base_url}")
        return False
    except Exception as e:
        print(f"  [FAIL] Health check error: {e}")
        return False


def run_audit(
    base_url: str,
    api_key: str,
    model_name: str,
    model_path: str,
    data_path: str,
    label_col: str,
    timeout: int,
) -> Optional[dict]:
    """
    Upload model + validation data and trigger a full audit scan.
    POST /api/v1/audit/run  (multipart/form-data)

    Returns the JSON response or None on failure.
    """
    print(f"\n  Submitting audit scan for '{model_name}'...")

    if not os.path.exists(model_path):
        print(f"  [FAIL] Model file not found: {model_path}")
        return None

    if not os.path.exists(data_path):
        print(f"  [FAIL] Data file not found: {data_path}")
        return None

    try:
        with open(model_path, "rb") as mf, \
             open(data_path, "rb") as df:
            files = {
                "model_file": (
                    os.path.basename(model_path),
                    mf,
                    "application/octet-stream",
                ),
                "val_file": (
                    os.path.basename(data_path),
                    df,
                    "text/csv",
                ),
            }
            form_data = {
                "model_name": model_name,
                "label_col": label_col,
                "selected": [
                    "drift",
                    "performance",
                    "fairness",
                    "security",
                ],
            }
            r = requests.post(
                f"{base_url}/api/v1/audit/run",
                headers=get_headers(api_key),
                files=files,
                data=form_data,
                timeout=timeout,
            )

        if r.status_code not in (200, 202):
            print(f"  [FAIL] Audit rejected: HTTP {r.status_code}")
            print(f"    Response: {r.text[:400]}")
            return None

        result = r.json()
        print(f"  [OK] Audit accepted"
              f" (job_id={result.get('job_id', 'N/A')})")
        return result

    except requests.Timeout:
        print(f"  [FAIL] Audit request timed out after {timeout}s")
        return None
    except Exception as e:
        print(f"  [FAIL] Audit error: {e}")
        return None


def write_result(result: dict):
    """
    Write governance_result.json for the PR comment step
    to consume via actions/github-script.
    """
    output_path = "governance_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Result written -> {output_path}")


def poll_gate_result(base_url: str, api_key: str, submission_token: str, timeout: int) -> dict:
    import time
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            print(f"\n[!!] CI GATE TIMEOUT — job did not complete within {timeout}s")
            sys.exit(2)
            
        try:
            r = requests.get(
                f"{base_url}/api/v1/gate/result/{submission_token}",
                headers=get_headers(api_key),
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "COMPLETED":
                    return data
                elif data.get("status") == "FAILED":
                    print(f"\n[!!] CI GATE FAILED: {data.get('error')}")
                    sys.exit(2)
                else:
                    eta = data.get("eta_seconds", 15)
                    print(f"  [WAIT] Job running. ETA ~{eta}s. Polling in 5s...")
            else:
                print(f"\n[!!] API Error polling gate result: HTTP {r.status_code}")
                sys.exit(2)
        except Exception as e:
            print(f"\n[!!] Request error while polling: {e}")
            sys.exit(2)
            
        time.sleep(5)

def main():
    args = parse_args()
    strict = args.strict.lower() == "true"

    print(SEP)
    print("   ML Guard CI/CD Governance Gate  v8.0")
    print(SEP)
    print(f"   Model     : {args.model_name}")
    print(f"   Min Score : {args.min_score}")
    print(f"   Strict    : {strict}")
    print(f"   API URL   : {args.api_url}")

    print("\n[1/3] API Health Check")
    if not check_health(args.api_url, args.api_key):
        print("\n  [!!] ML Guard API is unreachable.")
        sys.exit(2)

    print("\n[2/3] Governance Audit Scan Submission")
    audit_result = run_audit(
        base_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model_name,
        model_path=args.model_path,
        data_path=args.data_path,
        label_col=args.label_col,
        timeout=args.timeout,
    )
    if not audit_result or "submission_token" not in audit_result:
        print("  [!!] Failed to get submission_token from audit.")
        sys.exit(2)
        
    submission_token = audit_result["submission_token"]
    print(f"  [OK] Submission Token: {submission_token}")

    print("\n[3/3] Polling Gate Result")
    gate_result = poll_gate_result(args.api_url, args.api_key, submission_token, args.timeout)
    
    score = gate_result.get("score", 0)
    verdict = gate_result.get("verdict", "FAILED")
    breach_count = gate_result.get("breach_count", 0)
    
    passed = False
    if score >= args.min_score:
        if verdict == "CERTIFIED":
            passed = True
        elif verdict == "CONDITIONAL" and not strict:
            passed = True

    write_result({
        "model_name": args.model_name,
        "model_id": gate_result.get("model_id"),
        "overall_score": score,
        "verdict": verdict,
        "gate_passed": passed,
        "contract_breaches": breach_count,
        "passed": passed,
    })
    
    print("\n" + SEP)
    print(f"   Score   : {score}/100")
    print(f"   Verdict : {verdict}")
    print(f"   Breaches: {breach_count}")
    print(SEP)
    
    if passed:
        print("   [OK] GOVERNANCE GATE PASSED")
        sys.exit(0)
    else:
        print("   [FAIL] GOVERNANCE GATE FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
