# ML Guard GitHub Actions Setup

## Required GitHub Secrets

Go to your repo → **Settings → Secrets and variables → Actions → Secrets**
and add:

| Secret | Value | Required |
|--------|-------|----------|
| `ML_GUARD_API_KEY` | Your API key (`mlg_...`) | ✅ Yes |
| `ML_GUARD_API_URL` | `https://your-mlguard.onrender.com` | ✅ Yes |

## Required GitHub Variables

Go to **Settings → Secrets and variables → Actions → Variables**
and add:

| Variable | Value |
|----------|-------|
| `ML_GUARD_API_URL` | `https://your-mlguard.onrender.com` |

> **Why both?**
> The `governance-gate` job uses `vars.ML_GUARD_API_URL` in its `if:` condition
> (secrets can't be evaluated in `if:` expressions) and `secrets.ML_GUARD_API_URL`
> for the actual request. Both must be set.

---

## How the Pipeline Works

```
Push / PR
    │
    ├── [Job 1] Backend Integrity Check
    │     ├── Install minimal deps
    │     ├── Verify all Python imports load correctly
    │     └── Verify all 16 router files exist
    │
    ├── [Job 2] Unit Tests          (needs: backend-check)
    │     ├── GovernanceEngine verdict thresholds
    │     ├── Live decay formula
    │     ├── CertificateEngine hash determinism
    │     └── ContractEngine operator evaluation
    │
    ├── [Job 3] Governance Gate     (needs: unit-tests)
    │     │     SKIPPED if secrets not set
    │     ├── Health check → /health
    │     ├── Audit scan  → POST /api/v1/audit/run
    │     ├── Gate check  → POST /api/v1/governance/{id}/gate
    │     └── PR comment  → posts governance table to the PR
    │
    └── [Job 4] Frontend Build Check  (independent, always runs)
          ├── npm ci
          ├── TypeScript check (non-blocking)
          └── next build
```

---

## Gate Behavior

| Condition | Exit Code | Pipeline |
|-----------|-----------|----------|
| Score ≥ 60 **AND** gate passes | `0` | ✅ Proceeds |
| Score < 60 **OR** gate fails | `1` | ❌ Blocked |
| API unreachable | `0` | ✅ Non-blocking |
| Strict mode ON + verdict ≠ CERTIFIED | `1` | ❌ Blocked |
| Secrets not set | Job skipped | ✅ Non-blocking |

### Strict Mode

Pass `--strict true` (or set the `strict_mode` input) to require a **CERTIFIED**
verdict in addition to passing the score threshold. Useful for production-branch
gates where CONDITIONAL is not acceptable.

---

## Running Locally

```bash
# From the repo root
python .github/scripts/ml_guard_ci.py \
  --api-url http://127.0.0.1:8000 \
  --api-key mlg_your_key \
  --model-name TestChurnModel-v1 \
  --model-path ml_guard/backend/fair_loan_model.pkl \
  --data-path ml_guard/backend/fair_loan_test.csv \
  --label-col target \
  --min-score 60

# Strict mode (require CERTIFIED verdict)
python .github/scripts/ml_guard_ci.py \
  --api-url http://127.0.0.1:8000 \
  --api-key mlg_your_key \
  --model-name TestChurnModel-v1 \
  --model-path ml_guard/backend/fair_loan_model.pkl \
  --data-path ml_guard/backend/fair_loan_test.csv \
  --min-score 70 \
  --strict true
```

---

## Triggering Manually

In the GitHub UI:

1. Go to **Actions → ML Guard — Governance Gate**
2. Click **Run workflow**
3. Optionally override:
   - `Model name to audit` (default: `TestChurnModel-v1`)
   - `Fail on WARNING (strict)` (default: `false`)

---

## PR Comment Preview

When the gate runs on a pull request, it automatically posts a comment:

```
## ML Guard Governance Report

✅ Verdict: CERTIFIED

| Metric        | Value        |
|---------------|--------------|
| Overall Score | 84/100       |
| Live Score    | 81/100       |
| Gate Status   | PASSED ✅    |

### Recommendations
- Monitor PSI on feature 'age' — approaching threshold
```

---

## Header Note (Bug Fix)

The original CI script used `X-Api-Key`. The correct header is **`X-API-Key`**.
This has been fixed in `.github/scripts/ml_guard_ci.py`. If you have any other
scripts or SDK calls, ensure they use `X-API-Key`.

---

## File Reference

| File | Purpose |
|------|---------|
| `.github/workflows/ml-governance.yml` | GitHub Actions workflow (4 jobs) |
| `.github/scripts/ml_guard_ci.py` | CI gate script (run locally or in CI) |
| `.github/SETUP.md` | This file |
