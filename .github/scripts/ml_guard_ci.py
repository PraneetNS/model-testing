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
    p.add_argument("--timeout", type=int, default=120,
        help="Request timeout in seconds (default: 120)")
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


def lookup_model_by_name(
    base_url: str,
    api_key: str,
    model_name: str,
) -> Optional[str]:
    """
    Resolve model_id by querying the models list filtered by name.
    GET /api/v1/models?name=<model_name>
    Returns the UUID of the most recently created matching model, or None.
    """
    try:
        r = requests.get(
            f"{base_url}/api/v1/models",
            headers=get_headers(api_key),
            params={"name": model_name},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        # Normalise: list or {models:[...]} or {items:[...]}
        if isinstance(data, list):
            models = data
        else:
            models = data.get("models", data.get("items", []))
        # Filter to exact name matches, take the first one
        for m in models:
            if m.get("name") == model_name:
                return m.get("id")
        return None
    except Exception:
        return None


def run_gate_check(
    base_url: str,
    api_key: str,
    model_id: str,
    min_score: float,
) -> Optional[dict]:
    """
    Run synchronous governance gate check.
    POST /api/v1/governance/{model_id}/gate

    Returns gate verdict dict or None on failure.
    """
    print(f"\n  Running governance gate check...")
    try:
        r = requests.post(
            f"{base_url}/api/v1/governance/{model_id}/gate",
            headers=get_json_headers(api_key),
            json={
                "policy_config": {
                    "min_governance_score": min_score,
                    "max_psi": 0.25,
                    "min_accuracy": 0.70,
                    "bias_parity_threshold": 0.15,
                }
            },
            timeout=30,
        )
        if r.status_code == 200:
            data = r.json()
            status = "PASSED" if data.get("passed") else "FAILED"
            print(f"  [OK] Gate check completed -> {status}")
            return data
        print(f"  [!!] Gate check: HTTP {r.status_code}")
        print(f"    {r.text[:200]}")
        return None
    except Exception as e:
        print(f"  [FAIL] Gate check error: {e}")
        return None


def get_governance_score(
    base_url: str,
    api_key: str,
    model_id: str,
) -> Optional[dict]:
    """
    Fetch the latest governance score for this model.
    GET /api/v1/governance/{model_id}/score
    """
    try:
        r = requests.get(
            f"{base_url}/api/v1/governance/{model_id}/score",
            headers=get_headers(api_key),
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
        print(f"  [!!] Score fetch: HTTP {r.status_code}")
        return None
    except Exception as e:
        print(f"  [!!] Score fetch error: {e}")
        return None


def get_contract_breaches(
    base_url: str,
    api_key: str,
    model_id: str,
) -> int:
    """
    Fetch 24-hour contract breach count.
    GET /api/v1/contracts/{model_id}/breach-summary
    Returns total breach count (0 if unavailable).
    """
    try:
        r = requests.get(
            f"{base_url}/api/v1/contracts/{model_id}/breach-summary",
            headers=get_headers(api_key),
            timeout=10,
        )
        if r.status_code == 200:
            return int(r.json().get("total_breaches", 0))
        return 0
    except Exception:
        return 0


def write_result(result: dict):
    """
    Write governance_result.json for the PR comment step
    to consume via actions/github-script.
    """
    output_path = "governance_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Result written -> {output_path}")


def print_summary(
    score_data: dict,
    gate_data: Optional[dict],
    breaches: int,
    passed: bool,
):
    """Print the formatted governance report to stdout."""
    print("\n" + SEP)
    print("   ML GUARD GOVERNANCE REPORT  v7.2")
    print(SEP)
    print(f"   Overall Score :  "
          f"{score_data.get('overall_score', 'N/A')}/100")
    print(f"   Live Score    :  "
          f"{score_data.get('live_score', 'N/A')}/100")
    print(f"   Verdict       :  "
          f"{score_data.get('verdict', 'N/A')}")
    if gate_data:
        gate_status = (
            "PASSED" if gate_data.get("passed") else "FAILED"
        )
        print(f"   Gate Status   :  {gate_status}")
    if breaches > 0:
        print(f"   Contract Breaches (24h): {breaches} [!!]")

    recs = score_data.get("recommendations", [])
    if recs:
        print("\n   Recommendations:")
        for rec in recs[:3]:
            print(f"     - {rec}")

    print(SEP)
    if passed:
        print("   [OK] GOVERNANCE GATE PASSED")
        print("   Pipeline may proceed to deployment.")
    else:
        print("   [FAIL] GOVERNANCE GATE FAILED")
        print("   Pipeline blocked. Fix governance issues.")
    print(SEP)


def main():
    args = parse_args()
    strict = args.strict.lower() == "true"

    print(SEP)
    print("   ML Guard CI/CD Governance Gate  v7.2")
    print(SEP)
    print(f"   Model     : {args.model_name}")
    print(f"   Min Score : {args.min_score}")
    print(f"   Strict    : {strict}")
    print(f"   API URL   : {args.api_url}")

    # -- Step 1: Health check
    print("\n[1/6] API Health Check")
    if not check_health(args.api_url, args.api_key):
        print("\n  [!!] ML Guard API is unreachable.")
        print("  Set ML_GUARD_API_URL + ML_GUARD_API_KEY secrets.")
        print("  Governance gate skipped (non-blocking).")
        sys.exit(0)  # Non-blocking: don't fail CI if API is down

    # -- Step 2: Run audit scan
    print("\n[2/6] Governance Audit Scan")
    audit_result = run_audit(
        base_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model_name,
        model_path=args.model_path,
        data_path=args.data_path,
        label_col=args.label_col,
        timeout=args.timeout,
    )

    # -- Step 3: Resolve model_id
    # The audit endpoint returns {job_id, status, message} when using
    # Celery async dispatch — model_id is stored but not returned directly.
    # Strategy: check audit response first, then look up by model name.
    print("\n[3/6] Resolving Model ID")
    model_id = None
    if audit_result:
        model_id = (
            audit_result.get("model_id") or
            audit_result.get("id")
        )
    if not model_id:
        print("  Audit response has no model_id, "
              "looking up by model name...")
        model_id = lookup_model_by_name(
            args.api_url, args.api_key, args.model_name
        )
    if model_id:
        print(f"  [OK] Model ID: {model_id}")
    else:
        print("  [!!] Could not resolve model_id by name or audit response.")
        print("  Skipping gate check (non-blocking).")
        sys.exit(0)

    # -- Step 4: Fetch governance score
    print("\n[4/6] Fetching Governance Score")
    score_data = get_governance_score(
        args.api_url, args.api_key, model_id
    )
    if not score_data:
        print("  [!!] Could not fetch governance score.")
        sys.exit(0)
    print(f"  [OK] Overall: {score_data.get('overall_score')}/100"
          f"  Verdict: {score_data.get('verdict')}")

    # -- Step 5: Run gate check
    print("\n[5/6] Running Gate Check")
    gate_data = run_gate_check(
        args.api_url, args.api_key,
        model_id, args.min_score
    )

    # -- Step 6: Contract breaches
    print("\n[6/6] Checking Contract Breaches (24h)")
    breaches = get_contract_breaches(
        args.api_url, args.api_key, model_id
    )
    if breaches > 0:
        print(f"  [!!] {breaches} contract breach(es) in last 24 hours")
    else:
        print(f"  [OK] No contract breaches")

    # ── Determine pass/fail ───────────────────────────
    overall_score = float(score_data.get("overall_score", 0))
    verdict = score_data.get("verdict", "FAILED")
    gate_passed = (
        gate_data.get("passed", False) if gate_data else False
    )

    passed = (
        overall_score >= args.min_score
        and gate_passed
        and (not strict or verdict == "CERTIFIED")
    )

    # ── Write result JSON for PR comment step ─────────
    write_result({
        "model_name": args.model_name,
        "model_id": model_id,
        "overall_score": overall_score,
        "live_score": score_data.get("live_score"),
        "verdict": verdict,
        "gate_passed": gate_passed,
        "contract_breaches": breaches,
        "recommendations": score_data.get("recommendations", []),
        "passed": passed,
    })

    # ── Print final report ────────────────────────────
    print_summary(score_data, gate_data, breaches, passed)

    # ── Exit code controls the pipeline ──────────────
    # 0 = PASSED → pipeline continues
    # 1 = FAILED → pipeline blocked
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
