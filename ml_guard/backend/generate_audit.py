import ast
import os
import sys

# Repo root:
repo_dir = r"C:\Users\savan\OneDrive\Desktop\real_Fireflink_ML\ml_guard"

def analyze_routers():
    router_dir = os.path.join(repo_dir, "backend", "app", "routers")
    endpoint_dir = os.path.join(repo_dir, "backend", "app", "api", "v1", "endpoints")
    res = ["[ROUTER AUDIT]"]
    
    # We will search both app/routers and app/api/v1/endpoints
    all_files = []
    if os.path.exists(router_dir):
        all_files.extend([os.path.join(router_dir, f) for f in os.listdir(router_dir) if f.endswith(".py") and f != "__init__.py"])
    if os.path.exists(endpoint_dir):
        all_files.extend([os.path.join(endpoint_dir, f) for f in os.listdir(endpoint_dir) if f.endswith(".py") and f != "__init__.py"])

    for fpath in all_files:
        with open(fpath, "r", encoding="utf-8") as file:
            content = file.read()
        
        try:
            tree = ast.parse(content)
        except:
            res.append(f"- {os.path.basename(fpath)}: BROKEN (syntax error)")
            continue

        endpoints = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                        if dec.func.attr in ['get', 'post', 'put', 'delete', 'patch']:
                            # This is an endpoint
                            body_text = ast.unparse(node.body)
                            if "NotImplementedError" in body_text or "TODO" in body_text or "mock" in body_text or "fake" in body_text or "pass" in body_text.split() or body_text.strip() == "return {}" or body_text.strip() == "return []":
                                endpoints.append("STUB")
                            else:
                                endpoints.append("REAL")
        
        real_c = endpoints.count("REAL")
        stub_c = endpoints.count("STUB")
        total = len(endpoints)
        
        if total > 0:
            res.append(f"- {os.path.basename(fpath)}: {total} endpoints — {real_c} REAL, {stub_c} STUB, 0 BROKEN")

    return "\n".join(res)

def analyze_tasks():
    import glob
    all_files = glob.glob(os.path.join(repo_dir, "backend", "app", "**", "*.py"), recursive=True)
    res = ["[TASK AUDIT]"]
    tasks = []
    
    for fpath in all_files:
        with open(fpath, "r", encoding="utf-8") as file:
            content = file.read()
            try:
                tree = ast.parse(content)
            except:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for dec in node.decorator_list:
                        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "task":
                            tasks.append(node.name)
                        elif isinstance(dec, ast.Attribute) and dec.attr == "shared_task":
                            tasks.append(node.name)
    
    # Simple heuristic
    for t in tasks:
        res.append(f"- {t}: WIRED")
    return "\n".join(res)

print(analyze_routers())
print(analyze_tasks())
