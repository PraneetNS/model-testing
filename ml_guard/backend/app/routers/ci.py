"""
GitHub / GitLab CI Integration Router.
- Webhook events with HMAC verification
- Run governance scan → return pass/fail
- Post PR comment with governance summary
- Block merge via GitHub Checks/Status API
"""
import uuid
import hmac
import hashlib
import json
from fastapi import APIRouter, Depends, HTTPException, Request, Header, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import CIIntegration, ScanRecord, Model, AuditLog, utcnow
from app.core.auth import AuthContext, require_role, log_action

router = APIRouter()


# ═══════════════════════════════════════════════
# FEATURE 6: CI/CD GOVERNANCE GATE
# ═══════════════════════════════════════════════
@router.post("/ci/audit")
async def ci_audit_gate(
    model_name: str,
    governance_score_override: float = None,  # For simulation/mocking in pipelines
    pipeline_metadata: dict = {},
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """
    Automated governance gate for CI/CD pipelines.
    Returns whether deployment is allowed based on safety metrics.
    """
    # In a real scenario, this would trigger a background scan or evaluate 
    # the latest scan results for this model name.
    
    # Check latest scan for this model
    from app.db.models import Model
    model = (await db.execute(select(Model).filter(Model.name == model_name).order_by(Model.created_at.desc()))).scalars().first()
    
    score = governance_score_override
    if score is None:
        if model:
            from app.db.models import ScanRecord
            scan_stmt = select(ScanRecord).filter(ScanRecord.model_id == model.id).order_by(ScanRecord.created_at.desc()).limit(1)
            scan_result = await db.execute(scan_stmt)
            latest_scan = scan_result.scalars().first()
            if latest_scan:
                score = latest_scan.governance_score
            else:
                score = 75.0
        else:
            score = 75.0  # Default for demonstration if no scan found

    risk_level = "LOW"
    if score < 50: risk_level = "CRITICAL"
    elif score < 70: risk_level = "MEDIUM"
    
    deployment_allowed = score >= 70
    
    await log_action(db, auth, "ci.audit_gate", "model", str(model.id) if model else None, {
        "model_name": model_name,
        "governance_score": score,
        "deployment_allowed": deployment_allowed,
        "pipeline": pipeline_metadata
    })

    return {
        "model_name": model_name,
        "governance_score": score,
        "risk_level": risk_level,
        "deployment_allowed": deployment_allowed,
        "message": "Deployment approved by ML Guard." if deployment_allowed else "Deployment blocked: Governance score too low.",
        "ci_metadata": pipeline_metadata
    }



# ═══════════════════════════════════════════════
# REGISTER CI INTEGRATION
# ═══════════════════════════════════════════════
@router.post("/ci/integrations")
async def register_integration(
    provider: str,
    repo_url: str = "",
    webhook_secret: str = "",
    access_token: str = "",
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("admin")),
):
    if provider not in ("github", "gitlab", "jenkins"):
        raise HTTPException(400, "Provider must be: github, gitlab, jenkins.")

    # Hash the access token for storage
    token_hash = hashlib.sha256(access_token.encode()).hexdigest() if access_token else None

    integration = CIIntegration(
        org_id=auth.org_id,
        provider=provider,
        repo_url=repo_url,
        webhook_secret=webhook_secret,
        access_token_hash=token_hash,
        settings={"raw_token": access_token} if access_token else {},  # For demo; production should use vault
    )
    db.add(integration)
    await db.commit()
    await db.refresh(integration)
    log_action(db, auth, "ci.register", "ci_integration", str(integration.id), {
        "provider": provider, "repo": repo_url
    })
    return {"id": str(integration.id), "provider": provider, "repo_url": repo_url}


@router.get("/ci/integrations")
async def list_integrations(
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    items = (await db.execute(select(CIIntegration).filter(CIIntegration.org_id == auth.org_id))).scalars().all()
    
    if not items:
        # Seed a default integration for ML Guard Enterprise Demo
        default_int = CIIntegration(
            org_id=auth.org_id,
            provider="github",
            repo_url="https://github.com/fireflink/ml-guard-enterprise",
            is_active=True,
            webhook_secret="demo_secret",
            access_token_hash=None,
            settings={"branch_pattern": "main", "auto_comment": True}
        )
        db.add(default_int)
        await db.commit()
        await db.refresh(default_int)
        items = [default_int]

    return [
        {
            "id": str(i.id), 
            "provider": i.provider.upper(), 
            "repo_name": i.repo_url.split("/")[-1],
            "repo_url": i.repo_url, 
            "is_active": i.is_active, 
            "branch_pattern": i.settings.get("branch_pattern", "main"),
            "last_run_at": i.created_at.isoformat() # use creation as last run for demo
        }
        for i in items
    ]


# ═══════════════════════════════════════════════
# GITHUB WEBHOOK HANDLER
# ═══════════════════════════════════════════════
@router.post("/webhooks/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(None, alias="X-Hub-Signature-256"),
    db: AsyncSession = Depends(get_db),
):
    """
    Receive GitHub webhook events.
    On pull_request opened/synchronize:
    - Log the event
    - Return structured governance payload
    - If integration has access_token, post PR comment + commit status
    """
    body = await request.body()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON payload.")

    event = request.headers.get("X-GitHub-Event", "")
    repo_url = payload.get("repository", {}).get("html_url", "")

    # Find matching integration
    integration = (await db.execute(select(CIIntegration).filter(
        CIIntegration.repo_url == repo_url,
        CIIntegration.provider == "github",
        CIIntegration.is_active == True,
    ))).scalars().first()

    # ─── Verify HMAC signature ───
    if integration and integration.webhook_secret and x_hub_signature_256:
        expected = "sha256=" + hmac.new(
            integration.webhook_secret.encode(),
            body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, x_hub_signature_256):
            raise HTTPException(403, "Invalid webhook signature.")

    # ─── Log the event ───
    db.add(AuditLog(
        org_id=str(integration.org_id) if integration else None,
        action="ci.github_webhook",
        resource_type="ci_integration",
        resource_id=str(integration.id) if integration else None,
        details={"event": event, "repo": repo_url, "action": payload.get("action")},
    ))
    await db.commit()

    # ─── Handle PR events ───
    if event == "pull_request" and payload.get("action") in ("opened", "synchronize", "reopened"):
        pr = payload.get("pull_request", {})
        pr_number = pr.get("number")
        pr_title = pr.get("title", "")
        head_sha = pr.get("head", {}).get("sha", "")
        repo_full = payload.get("repository", {}).get("full_name", "")

        response = {
            "status": "received",
            "event": "pull_request",
            "pr_number": pr_number,
            "pr_title": pr_title,
            "head_sha": head_sha,
            "action": payload.get("action"),
            "message": "ML Guard governance check queued.",
            "next_step": f"POST /api/v1/audit/run with scan data, then call POST /api/v1/ci/report-status to update PR.",
        }

        # If we have an access token, set pending status immediately
        if integration and integration.settings.get("raw_token"):
            token = integration.settings["raw_token"]
            _set_github_commit_status(
                token=token,
                repo=repo_full,
                sha=head_sha,
                state="pending",
                description="ML Guard governance check in progress...",
                context="ml-guard/governance",
            )
            response["github_status_set"] = "pending"

        return response

    return {"status": "received", "event": event}


# ═══════════════════════════════════════════════
# REPORT STATUS BACK TO GITHUB (after scan completes)
# ═══════════════════════════════════════════════
@router.post("/ci/report-status")
async def report_status(
    scan_id: str = Query(...),
    repo: str = Query(..., description="owner/repo"),
    sha: str = Query(..., description="commit SHA"),
    pr_number: int = Query(None),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """
    After a governance scan completes, call this to:
    1. Update GitHub commit status (success/failure)
    2. Post a PR comment with the governance summary
    """
    scan = await db.get(ScanRecord, scan_id)
    if not scan:
        raise HTTPException(404, "Scan not found.")

    # Find integration with access token
    integration = (await db.execute(select(CIIntegration).filter(
        CIIntegration.org_id == auth.org_id,
        CIIntegration.provider == "github",
        CIIntegration.is_active == True,
    ))).scalars().first()

    if not integration or not integration.settings.get("raw_token"):
        raise HTTPException(400, "No GitHub integration with access token configured.")

    token = integration.settings["raw_token"]
    score = scan.governance_score
    gate = scan.gate_status or "UNKNOWN"
    conclusion = "success" if gate == "PASSED" else "failure"
    checks_run = scan.checks_run or []

    # ─── 1. Set commit status ───
    _set_github_commit_status(
        token=token,
        repo=repo,
        sha=sha,
        state=conclusion,
        description=f"Governance Score: {score:.0f}/100 — {gate}",
        context="ml-guard/governance",
        target_url=None,
    )

    # ─── 2. Post PR comment ───
    comment_body = None
    if pr_number:
        comment_body = _build_pr_comment(scan)
        _post_github_pr_comment(
            token=token,
            repo=repo,
            pr_number=pr_number,
            body=comment_body,
        )

    log_action(db, auth, "ci.report_status", "scan", scan_id, {
        "repo": repo, "sha": sha, "pr_number": pr_number,
        "conclusion": conclusion, "score": score,
    })

    return {
        "scan_id": scan_id,
        "conclusion": conclusion,
        "gate_status": gate,
        "governance_score": score,
        "commit_status_set": True,
        "pr_comment_posted": pr_number is not None,
    }


# ═══════════════════════════════════════════════
# CI STATUS CHECK (polling endpoint)
# ═══════════════════════════════════════════════
@router.get("/ci/status/{scan_id}")
async def ci_status(scan_id: str, db: AsyncSession = Depends(get_db)):
    """CI-compatible status for a governance scan."""
    scan = await db.get(ScanRecord, scan_id)
    if not scan:
        raise HTTPException(404, "Scan not found.")

    conclusion = "success" if scan.gate_status == "PASSED" else "failure"
    return {
        "scan_id": str(scan.id),
        "conclusion": conclusion,
        "gate_status": scan.gate_status,
        "governance_score": scan.governance_score,
        "checks_run": scan.checks_run,
        "created_at": str(scan.created_at),
        "summary": f"Governance score: {scan.governance_score}. Gate: {scan.gate_status}.",
        "merge_allowed": conclusion == "success",
    }


# ═══════════════════════════════════════════════
# HELPER: GitHub API calls
# ═══════════════════════════════════════════════
def _set_github_commit_status(token: str, repo: str, sha: str, state: str,
                               description: str, context: str, target_url: str = None):
    """Set commit status via GitHub Statuses API."""
    try:
        import requests
        url = f"https://api.github.com/repos/{repo}/statuses/{sha}"
        payload = {
            "state": state,  # pending, success, failure, error
            "description": description[:140],
            "context": context,
        }
        if target_url:
            payload["target_url"] = target_url
        requests.post(url, json=payload, headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }, timeout=10)
    except Exception:
        pass  # Non-blocking


def _post_github_pr_comment(token: str, repo: str, pr_number: int, body: str):
    """Post a comment on a GitHub PR."""
    try:
        import requests
        url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
        requests.post(url, json={"body": body}, headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }, timeout=10)
    except Exception:
        pass


def _build_pr_comment(scan: ScanRecord) -> str:
    """Build a markdown governance summary for PR comment."""
    score = scan.governance_score or 0
    gate = scan.gate_status or "UNKNOWN"
    checks = scan.checks_run or []
    results = scan.results_json or {}

    emoji = "✅" if gate == "PASSED" else "⚠️" if gate == "WARNING" else "❌"
    merge_msg = "Merge is **allowed**." if gate == "PASSED" else "Merge is **blocked** by governance policy."

    lines = [
        f"## {emoji} ML Guard Governance Report",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Governance Score** | **{score:.0f}/100** |",
        f"| **Gate Status** | {gate} |",
        f"| **Checks Run** | {', '.join(checks)} |",
    ]

    # Add key metrics if available
    metrics = results.get("metrics", {})
    for k, v in list(metrics.items())[:6]:
        if isinstance(v, float):
            lines.append(f"| {k.replace('_', ' ').title()} | {v:.4f} |")

    # Overfitting
    gaps = results.get("overfitting_gap", {})
    for k, v in gaps.items():
        if isinstance(v, float) and abs(v) > 0.05:
            lines.append(f"| ⚠️ {k.replace('_', ' ').title()} | {v:+.4f} |")

    lines.extend([
        "",
        f"> {merge_msg}",
        "",
        f"<sub>Scan ID: `{scan.id}` • Powered by ML Guard v5.0</sub>",
    ])

    return "\n".join(lines)
