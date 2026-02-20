
import os
import shutil
from pathlib import Path

def cleanup_workspace():
    root = Path.cwd()
    legacy_dir = root / "legacy_v1"
    
    # Create legacy directory if it doesn't exist
    legacy_dir.mkdir(exist_ok=True)
    
    # Items to move (everything except legacy_v1 and the cleanup script itself/hidden files)
    # We want to keep .git if it exists, and .gitignore
    
    for item in root.iterdir():
        if item.name == "legacy_v1" or item.name == "cleanup_script.py" or item.name.startswith("."):
            continue
            
        target = legacy_dir / item.name
        
        try:
            print(f"Moving {item.name} to {target}")
            if item.is_dir():
                # specific handling for directories to merge/overwrite if needed, 
                # but for now we assume legacy_v1 is empty or we can just move into it
                # shutil.move is generally robust
                shutil.move(str(item), str(target))
            else:
                shutil.move(str(item), str(target))
        except Exception as e:
            print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    cleanup_workspace()
