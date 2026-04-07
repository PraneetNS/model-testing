import subprocess, sys
result = subprocess.run(
    [sys.executable, "-u", "test_contracts.py"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    cwd=r"c:\Users\savan\OneDrive\Desktop\real_Fireflink_ML\ml_guard\backend"
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:500])
