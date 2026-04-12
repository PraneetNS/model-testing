from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request, HTTPException

def get_api_key_identifier(request: Request) -> str:
    """Identify request by API key for rate limiting."""
    return request.headers.get("X-API-Key", get_remote_address(request))

limiter = Limiter(
    key_func=get_api_key_identifier,
    default_limits=["120/minute"]
)
