# CI/CD Integration Guide

## GitHub Actions Integration

### 1. Register your repository

```bash
curl -X POST "https://mlguard.yourorg.com/api/v1/ci/integrations" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "github",
    "org_id": "<your-org-id>",
    "repo_url": "https://github.com/your-org/ml-models",
    "webhook_secret": "your-secret-here"
  }'
```

### 2. Add GitHub Actions Workflow

```yaml
# .github/workflows/governance-check.yml
name: ML Governance Check
on:
  pull_request:
    paths:
      - 'models/**'
      - 'data/**'

jobs:
  governance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Governance Scan
        run: |
          SCAN_RESULT=$(curl -s -X POST "https://mlguard.yourorg.com/api/v1/audit/run" \
            -F "model_file=@models/model.pkl" \
            -F "train_file=@data/train.csv" \
            -F "val_file=@data/val.csv" \
            -F "checks=accuracy,f1,psi_drift,overfitting_check,calibration_check" \
            -F "label_col=target" \
            -F "model_name=${{ github.sha }}")

          SCAN_ID=$(echo $SCAN_RESULT | jq -r '.scan_id')
          echo "SCAN_ID=$SCAN_ID" >> $GITHUB_ENV

      - name: Check Governance Gate
        run: |
          STATUS=$(curl -s "https://mlguard.yourorg.com/api/v1/ci/status/${{ env.SCAN_ID }}")
          CONCLUSION=$(echo $STATUS | jq -r '.conclusion')
          SCORE=$(echo $STATUS | jq -r '.governance_score')

          echo "Governance Score: $SCORE"
          echo "Conclusion: $CONCLUSION"

          if [ "$CONCLUSION" != "success" ]; then
            echo "::error::Governance check FAILED. Score: $SCORE"
            exit 1
          fi
```

### 3. Webhook (PR Auto-Comment)

Configure a webhook in GitHub:
- URL: `https://mlguard.yourorg.com/api/v1/webhooks/github`
- Content type: `application/json`
- Secret: Your webhook secret
- Events: Pull requests

---

## GitLab Integration

```yaml
# .gitlab-ci.yml
governance_check:
  stage: test
  script:
    - |
      SCAN=$(curl -s -X POST "$ML_GUARD_URL/api/v1/audit/run" \
        -F "model_file=@models/model.pkl" \
        -F "train_file=@data/train.csv" \
        -F "val_file=@data/val.csv" \
        -F "checks=accuracy,psi_drift,calibration_check" \
        -F "label_col=target")
      SCAN_ID=$(echo $SCAN | python3 -c "import sys,json; print(json.load(sys.stdin).get('scan_id',''))")
      STATUS=$(curl -s "$ML_GUARD_URL/api/v1/ci/status/$SCAN_ID")
      echo $STATUS | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d['conclusion']=='success' else 1)"
  only:
    changes:
      - models/**
```

---

## API Key Authentication

Generate an API key for CI:

```bash
curl -X POST "https://mlguard.yourorg.com/api/v1/orgs/<org_id>/api-keys" \
  -H "Content-Type: application/json" \
  -d '{"label": "GitHub Actions", "scopes": ["audit", "behavior"]}'
```

Store the returned key as `ML_GUARD_API_KEY` in your CI secrets.

---

## Alert Configuration

Set up alerts for governance failures:

```bash
curl -X POST "https://mlguard.yourorg.com/api/v1/alerts/rules" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Governance Score Drop",
    "condition": {"metric": "governance_score", "op": "<", "value": 70},
    "channels": ["webhook"],
    "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
  }'
```

After each scan, evaluate alerts:

```bash
curl -X POST "https://mlguard.yourorg.com/api/v1/alerts/evaluate/<scan_id>"
```
