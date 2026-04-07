"""
test_contracts.py — ML Guard Model Behavior Contract System Test

Exercises the full contract lifecycle:
  1. Create contract with 3 promises (confidence, latency, distribution)
  2. Ingest a prediction that violates 2 of them
  3. Query breach records
  4. Check breach summary + governance penalty
  5. Verify governance score reflects breach penalty
  6. Mark a breach as resolved
  7. Dry-run validate a clean prediction

Run with the backend running:
    python test_contracts.py
"""
import requests
import json
import time

BASE     = "http://127.0.0.1:8000"
API_KEY  = "mlg_1Ai7zfmfsB_GLaoNuKjOOopFh12xLzGy7SDqh7Kho1U"
MODEL_ID = "f9597635-5c66-4b17-9e4b-38e3fde81a53"
HEADERS  = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

SEP = "-" * 50


def p(label, value=None):
    print(f"  {label}: {value}" if value is not None else f"  {label}")


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ── 1. Create contract ─────────────────────────────────────────────────────────

def test_create_contract():
    section("STEP 1 — Create Behavioral Contract")
    r = requests.post(
        f"{BASE}/api/v1/contracts",
        headers=HEADERS,
        json={
            "name": "Production Safety Contract v1",
            "model_id": MODEL_ID,
            "version": "1.0",
            "description": "Core behavioral promises for churn model",
            "promises": [
                {
                    "name": "Confident predictions only",
                    "type": "output",
                    "metric": "prediction_proba",
                    "operator": "gte",
                    "threshold": 0.55,
                    "severity": "MEDIUM",
                    "action": "flag",
                },
                {
                    "name": "Low latency guarantee",
                    "type": "latency",
                    "metric": "latency_ms",
                    "operator": "lte",
                    "threshold": 500,
                    "severity": "HIGH",
                    "action": "alert",
                },
                {
                    "name": "Reasonable positive prediction rate",
                    "type": "distribution",
                    "metric": "prediction_rate",
                    "operator": "lte",
                    "threshold": 0.90,
                    "window_hours": 24,
                    "severity": "HIGH",
                    "action": "alert",
                },
            ],
        },
    )
    print(f"  POST /contracts -> {r.status_code}")
    if r.status_code == 201:
        d = r.json()
        p("contract_id", d["contract_id"])
        p("promises_count", d["promises_count"])
        p("status", d["status"])
        return d["contract_id"]
    else:
        print(f"  ERROR: {r.text[:400]}")
        return None


# ── 2. Ingest violating prediction ────────────────────────────────────────────

def test_ingest_violating_prediction():
    section("STEP 2 — Ingest Prediction That Violates Promises")
    print("  Sending prediction with proba=0.30 (<0.55) and latency=600ms (>500ms)")
    r = requests.post(
        f"{BASE}/api/v1/ingest/predict",
        headers=HEADERS,
        json={
            "model_id": MODEL_ID,
            "features": {"f1": 0.5, "f2": -0.2, "f3": 0.8},
            "prediction": "1",
            "prediction_proba": 0.30,   # VIOLATES confident predictions (>=0.55)
            "latency_ms": 600.0,         # VIOLATES latency guarantee (<=500ms)
            "environment": "production",
        },
    )
    print(f"  POST /ingest/predict -> {r.status_code}")
    p("response", r.json())

    # Also send a clean prediction
    print("\n  Sending clean prediction (proba=0.85, latency=120ms)...")
    r2 = requests.post(
        f"{BASE}/api/v1/ingest/predict",
        headers=HEADERS,
        json={
            "model_id": MODEL_ID,
            "features": {"f1": 0.1, "f2": 0.3, "f3": 0.6},
            "prediction": "0",
            "prediction_proba": 0.85,   # OK
            "latency_ms": 120.0,         # OK
            "environment": "production",
        },
    )
    print(f"  POST /ingest/predict (clean) -> {r2.status_code}")


# ── 3. Query breaches ──────────────────────────────────────────────────────────

def test_get_breaches():
    section("STEP 3 — Query Contract Breaches")
    r = requests.get(
        f"{BASE}/api/v1/contracts/{MODEL_ID}/breaches",
        headers=HEADERS,
        params={"hours": 1},
    )
    print(f"  GET /contracts/{MODEL_ID[:12]}../breaches -> {r.status_code}")
    if r.status_code == 200:
        breaches = r.json()
        p("total_breaches_found", len(breaches))
        for b in breaches[:5]:
            print(
                f"    [{b['severity']:8}] {b['promise_name']}: "
                f"expected {b['expected']}  got {b['actual']}"
            )
        return [b["breach_id"] for b in breaches]
    else:
        print(f"  ERROR: {r.text[:400]}")
        return []


# ── 4. Breach summary ──────────────────────────────────────────────────────────

def test_breach_summary():
    section("STEP 4 — Breach Summary + Governance Penalty")
    r = requests.get(
        f"{BASE}/api/v1/contracts/{MODEL_ID}/breach-summary",
        headers=HEADERS,
    )
    print(f"  GET /contracts/{MODEL_ID[:12]}../breach-summary -> {r.status_code}")
    if r.status_code == 200:
        d = r.json()
        p("total_breaches", d["total_breaches"])
        p("by_severity", d["by_severity"])
        p("by_promise", d["by_promise"])
        p("governance_penalty", f"-{d['governance_penalty']} pts")
        return d["governance_penalty"]
    else:
        print(f"  ERROR: {r.text[:400]}")
        return 0.0


# ── 5. Governance score ────────────────────────────────────────────────────────

def test_governance_score():
    section("STEP 5 — Governance Score (with contract penalty applied)")
    r = requests.get(
        f"{BASE}/api/v1/governance/{MODEL_ID}/score",
        headers=HEADERS,
    )
    print(f"  GET /governance/{MODEL_ID[:12]}../score -> {r.status_code}")
    if r.status_code == 200:
        d = r.json()
        p("overall_score", d.get("overall_score"))
        p("live_score",    d.get("live_score"))
        p("verdict",       d.get("verdict"))
        recs = d.get("recommendations", [])
        contract_recs = [rec for rec in recs if "contract" in rec.lower() or "breach" in rec.lower()]
        if contract_recs:
            print()
            print("  Contract-linked recommendations:")
            for rec in contract_recs:
                print(f"    !! {rec}")
        else:
            print("  (no contract breach recommendations yet)")


# ── 6. Resolve a breach ────────────────────────────────────────────────────────

def test_resolve_breach(breach_ids):
    if not breach_ids:
        return
    section("STEP 6 — Resolve a Breach")
    bid = breach_ids[0]
    r = requests.patch(
        f"{BASE}/api/v1/contracts/breaches/{bid}/resolve",
        headers=HEADERS,
    )
    print(f"  PATCH /contracts/breaches/{bid[:12]}../resolve -> {r.status_code}")
    if r.status_code == 200:
        p("result", r.json())


# ── 7. Dry-run validate ────────────────────────────────────────────────────────

def test_dry_run_validate():
    section("STEP 7 — Dry-Run Contract Validation")
    print("  Validating a high-confidence, low-latency prediction...")
    r = requests.post(
        f"{BASE}/api/v1/contracts/validate",
        headers=HEADERS,
        json={
            "model_id": MODEL_ID,
            "prediction": "0",
            "prediction_proba": 0.92,
            "features": {"f1": 0.1, "f2": 0.2, "f3": 0.3},
            "latency_ms": 45.0,
        },
    )
    print(f"  POST /contracts/validate -> {r.status_code}")
    if r.status_code == 200:
        d = r.json()
        p("compliant", d["compliant"])
        p("breach_count", d["breach_count"])
        if d["breaches"]:
            for b in d["breaches"]:
                print(f"    [{b['severity']}] {b['promise']}: actual={b['actual']} threshold={b['threshold']}")

    print("\n  Validating a low-confidence prediction (should breach)...")
    r2 = requests.post(
        f"{BASE}/api/v1/contracts/validate",
        headers=HEADERS,
        json={
            "model_id": MODEL_ID,
            "prediction": "1",
            "prediction_proba": 0.20,   # below 0.55 threshold
            "features": {"f1": 0.5, "f2": 0.5, "f3": 0.5},
            "latency_ms": 800.0,         # above 500ms threshold
        },
    )
    print(f"  POST /contracts/validate -> {r2.status_code}")
    if r2.status_code == 200:
        d2 = r2.json()
        p("compliant", d2["compliant"])
        p("breach_count", d2["breach_count"])
        for b in d2["breaches"]:
            print(f"    [{b['severity']}] {b['promise']}: actual={b['actual']} threshold={b['threshold']}")


# ── 8. List all contracts ──────────────────────────────────────────────────────

def test_list_contracts():
    section("STEP 8 — List All Contracts for Model")
    r = requests.get(
        f"{BASE}/api/v1/contracts/{MODEL_ID}",
        headers=HEADERS,
    )
    print(f"  GET /contracts/{MODEL_ID[:12]}.. -> {r.status_code}")
    if r.status_code == 200:
        contracts = r.json()
        p("contracts_found", len(contracts))
        for c in contracts:
            print(
                f"    [{c['version']}] {c['name']} "
                f"({'active' if c['is_active'] else 'inactive'}) "
                f"— {c['promises_count']} promises"
            )


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  ML Guard -- Contract System Test")
    print("=" * 50)

    contract_id = test_create_contract()

    if contract_id:
        test_ingest_violating_prediction()
        time.sleep(2.0)  # wait for background tasks to complete

        breach_ids = test_get_breaches()
        penalty = test_breach_summary()
        test_governance_score()
        test_resolve_breach(breach_ids)
        test_dry_run_validate()
        test_list_contracts()

    print(f"\n{'=' * 50}")
    print("  Contract Test Complete")
    print("=" * 50)
