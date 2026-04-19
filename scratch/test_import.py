import os
import sys

backend_dir = r"c:\Users\savan\OneDrive\Desktop\real_Fireflink_ML\ml_guard\backend"
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

try:
    from app.core.config import settings
    print("Success: Imported settings")
    print(f"Project Name: {settings.PROJECT_NAME}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
