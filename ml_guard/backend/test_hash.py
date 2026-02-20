from passlib.context import CryptContext
import traceback

try:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    h = pwd_context.hash("password123")
    print(f"Hash: {h}")
    v = pwd_context.verify("password123", h)
    print(f"Verify: {v}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
