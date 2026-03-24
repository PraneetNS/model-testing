from app.main import app
from fastapi.routing import APIRoute, APIWebSocketRoute

for route in app.routes:
    if isinstance(route, APIWebSocketRoute):
        print(f"WS ROUTE: {route.path}")
    elif isinstance(route, APIRoute):
        print(f"HTTP ROUTE: {route.path} [{','.join(route.methods)}]")
    else:
        print(f"OTHER: {route.path} ({type(route)})")
