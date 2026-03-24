from app.main import app

for route in app.routes:
    # Check for websocket routes
    if hasattr(route, "endpoint") and "websocket" in str(route.endpoint).lower():
        print(f"WS ROUTE: {route.path}")
    elif hasattr(route, "path"):
        print(f"ROUTE: {route.path}")
