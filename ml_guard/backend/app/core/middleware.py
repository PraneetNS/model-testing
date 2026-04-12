import re
import hashlib
import json
import structlog
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.db.models import SecurityAlert, APIKey
import asyncio

logger = structlog.get_logger()

# Injection Patterns
SQL_INJECTION_PATTERNS = [
    r"(?i)WAITFOR\s+DELAY",
    r"(?i)UNION\s+SELECT",
    r"(?i)OR\s+1=1",
    r"(?i)DROP\s+TABLE",
    r"(?i)INSERT\s+INTO",
    r"(?i)SELECT\s+.*\s+FROM",
]

PROMPT_INJECTION_PATTERNS = [
    r"(?i)ignore\s+previous\s+instructions",
    r"(?i)system:\s*override",
    r"(?i)you\s+are\s+now\s+an\s+admin",
    r"(?i)disregard\s+all\s+prior",
]

PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"%2e%2e/",
    r"\.\.\\",
]

class SecurityHardeningMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1. Scan JSON body for injection patterns
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    body_bytes = await request.body()
                    if body_bytes:
                        body_str = body_bytes.decode()
                        
                        # Pattern detection
                        detected = False
                        for pattern in SQL_INJECTION_PATTERNS + PROMPT_INJECTION_PATTERNS + PATH_TRAVERSAL_PATTERNS:
                            if re.search(pattern, body_str):
                                detected = True
                                break
                        
                        if detected:
                            await self.log_security_alert(request, body_str)
                            return Response(
                                content=json.dumps({"error": "Bad Request"}),
                                status_code=400,
                                media_type="application/json"
                            )
                        
                        # Replace request body so it can be read again by the endpoint
                        async def receive():
                            return {"type": "http.request", "body": body_bytes}
                        request._receive = receive

                except Exception as e:
                    logger.error("security_middleware_error", error=str(e))

        response = await call_next(request)

        # 2. Audit Logging (Background) for state-changing calls
        if request.method in ["POST", "PUT", "PATCH", "DELETE"]:
            # We skip some internal or health endpoints if needed
            if not request.url.path.startswith("/health"):
                asyncio.create_task(self.log_audit_event(request, response))

        return response

    async def log_audit_event(self, request: Request, response: Response):
        """Asynchronously log state-changing actions."""
        from app.db.session import SessionLocal
        from app.db.models import AuditLog
        
        client_ip = request.client.host if request.client else "unknown"
        endpoint = str(request.url.path)
        method = request.method
        status_code = response.status_code
        result = "success" if 200 <= status_code < 400 else "denied"
        
        # In a real app, we'd extract key_id from request.state if auth set it
        # For now, we'll just log what we have
        async with SessionLocal() as db:
            try:
                log = AuditLog(
                    action=f"{method}:{endpoint}",
                    actor_ip=client_ip,
                    result=result,
                    details={
                        "status_code": status_code,
                        "query_params": str(request.query_params)
                    }
                )
                db.add(log)
                await db.commit()
            except Exception as e:
                logger.error("audit_log_background_failed", error=str(e))

    async def log_security_alert(self, request: Request, body_str: str):
        # We need a DB session. Usually middleware doesn't have easy access to Depends(get_db)
        # So we'll get it manually or via background task
        payload_hash = hashlib.sha256(body_str.encode()).hexdigest()
        client_ip = request.client.host if request.client else "unknown"
        endpoint = str(request.url.path)
        x_api_key = request.headers.get("X-API-Key")
        
        # We'll use a one-off session
        from app.db.session import SessionLocal
        async with SessionLocal() as db:
            key_id = None
            if x_api_key:
                # Resolve key_id (bcrypt check or old sha256 check)
                # Since we are moving to bcrypt, we'll need a way to check both during migration
                # For now, let's just log what we can.
                from app.core.security import verify_password
                # This check might be slow, but it's an alert path
                # For now, we'll try to find the key by label or just skip resolving key_id in middleware 
                # to keep it fast, or look it up if we have to.
                
                # Simple lookup for now - in a real app we'd cache this
                # We'll just store the header or some identifier if not found
                pass

            alert = SecurityAlert(
                alert_type="injection_attempt",
                endpoint=endpoint,
                payload_hash=payload_hash,
                ip=client_ip,
                details={"method": request.method}
            )
            db.add(alert)
            await db.commit()
            logger.warning("security_alert_logged", endpoint=endpoint, ip=client_ip, hash=payload_hash)
