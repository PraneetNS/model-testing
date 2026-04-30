"""
Read-only full feature audit with corrected endpoint paths.
"""
import asyncio
import httpx

BASE = "http://localhost:8000"
H = {"X-API-Key": "dev-secret-key"}

# (method, feature_name, path, expected_ok_codes)
ENDPOINTS = [
    # Core health
    ("GET", "Health",            "/api/health",                       [200]),
    ("GET", "DB Health",         "/api/v1/health/db",                 [200]),
    # Auth
    ("GET", "API Keys",          "/api/v1/auth/apikeys",              [200]),
    # Registry / Inventory
    ("GET", "Inventory",         "/api/inventory",                    [200]),
    ("GET", "Models List",       "/api/v1/models",                    [200]),
    # Datasets
    ("GET", "Datasets",          "/api/v1/datasets",                  [200]),
    # Experiments
    ("GET", "Experiments",       "/api/v1/experiments",               [200]),
    # Deployments
    ("GET", "Deployments",       "/api/v1/deployments",               [200]),
    # Policies
    ("GET", "Active Policy",     "/api/v1/policies/active",           [200]),
    ("GET", "Policies List",     "/api/v1/policies",                  [200]),
    # Scan History
    ("GET", "Scan History",      "/api/v1/history",                   [200]),
    # Performance / Predictions
    ("GET", "Perf Stats",        "/api/v1/predictions/stats",         [200]),
    ("GET", "Perf Logs",         "/api/v1/predictions/logs?limit=5",  [200]),
    # Observability
    ("GET", "Observe Feed",      "/api/v1/observe/feed",              [200]),
    # Drift
    ("GET", "Drift Health",      "/api/v1/drift/health",              [200]),
    # Guardrail
    ("GET", "Guardrails",        "/api/guardrail",                    [200]),
    # Governance
    ("GET", "Governance Status", "/api/v1/governance/status",         [200]),
    ("GET", "Gov Trend",         "/api/v1/governance/trend",          [200]),
    # Compliance
    ("GET", "Compliance Packs",  "/api/compliance/packs/available",   [200]),
    # Alerts
    ("GET", "Alert Events",      "/api/v1/alerts/events?limit=5",     [200]),
    ("GET", "Alert Rules",       "/api/v1/alerts/rules",              [200]),
    ("GET", "Alert Summary",     "/api/v1/alerts/summary",            [200]),
    # Billing
    ("GET", "Billing Status",    "/api/billing/subscription",         [200]),
    ("GET", "Billing Usage",     "/api/billing/usage",                [200]),
    # CI
    ("GET", "CI Integrations",   "/api/v1/ci/integrations",          [200]),
    # Report Cards
    ("GET", "Report Cards",      "/api/v1/{model_id}/history".replace("{model_id}", "demo"),  [200, 404]),
    # Retraining
    ("GET", "Retraining Policy", "/api/v1/models/{model_id}/retraining-policy".replace("{model_id}", "demo"), [200, 404]),
    # Explainability
    ("GET", "Explainability",    "/api/v1/explainability/compute",    [405]),  # POST only
    # Security
    ("GET", "Security Alerts",   "/api/v1/security/alerts",          [200]),
    ("GET", "Security Stats",    "/api/v1/security/stats",           [200]),
    # Data Quality
    ("GET", "Data Quality",      "/api/v1/data-quality/validate",     [405]),  # POST only
    # Orgs
    ("GET", "Organizations",     "/api/v1/orgs",                     [200]),
    # Enterprise
    ("GET", "Enterprise Summary","/api/v1/enterprise/summary",       [200]),
    # Stream Drift
    ("GET", "Stream Models",     "/api/v1/stream/models",            [200]),
    # Advisory
    ("GET", "Advisory",          "/api/v1/advisory/explain",         [405]),  # POST only
    # Audit Log
    ("GET", "Audit Logs",        "/api/v1/audit-logs",               [200]),
    # Contracts
    ("GET", "Contracts",         "/api/v1/contracts",                [200]),
    # LLM Eval
    ("GET", "LLM History",       "/api/v1/llm/history",              [200]),
    # Preflight
    ("GET", "Preflight Health",  "/api/v1/preflight/health",         [200]),
    # Performance
    ("GET", "Perf Health",       "/api/v1/performance/health",       [200]),
]

async def audit():
    ok = warn = fail = 0
    rows = []
    async with httpx.AsyncClient(timeout=8.0) as client:
        for method, name, path, expected in ENDPOINTS:
            url = f"{BASE}{path}"
            try:
                r = await client.request(method, url, headers=H)
                code = r.status_code
                if code in expected:
                    status = "OK  "
                    ok += 1
                elif code in (200, 201, 204):
                    status = "OK  "
                    ok += 1
                elif code in (404, 405, 422):
                    status = "WARN"
                    warn += 1
                else:
                    status = "FAIL"
                    fail += 1
                try:
                    d = r.json()
                    if isinstance(d, list):
                        detail = f"list[{len(d)}]"
                    elif isinstance(d, dict):
                        keys = list(d.keys())[:4]
                        detail = f"{{{', '.join(str(k) for k in keys)}}}"
                    else:
                        detail = str(d)[:50]
                except Exception:
                    detail = r.text[:50]
            except Exception as e:
                code = 0
                status = "ERR "
                detail = str(e)[:50]
                fail += 1
            rows.append((status, code, name, detail))

    print("\n" + "="*82)
    print("  NIYANTRANA PLATFORM -- FULL FEATURE AUDIT")
    print("="*82)
    print(f"  {'ST':<6} {'HTTP':<6} {'FEATURE':<26} RESPONSE")
    print("-"*82)
    for status, code, name, detail in rows:
        marker = f"[{status}]"
        print(f"  {marker:<8} {code:<6} {name:<26} {detail}")
    print("="*82)
    print(f"  SUMMARY: {ok} OK  |  {warn} WARN  |  {fail} FAIL   (total {len(rows)})")
    print("="*82)

if __name__ == "__main__":
    asyncio.run(audit())
