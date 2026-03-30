import os
import sys
_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ml_guard_root = os.path.abspath(os.path.join(_backend_root, ".."))
_repo_root = os.path.abspath(os.path.join(_ml_guard_root, ".."))

sys.path.insert(0, _backend_root)
sys.path.insert(0, _repo_root)

try:
    from ml_guard.core import MLEvaluator
    print("MLEvaluator import SUCCESS")
except Exception as e:
    print(f"MLEvaluator import FAILED: {e}")

try:
    from ml_guard.core import evaluator
    print("evaluator import SUCCESS")
except Exception as e:
    print(f"evaluator import FAILED: {e}")
