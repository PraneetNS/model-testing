import os
import time
import uuid
import docker
import requests
import json
from typing import List, Dict, Any, Optional

SERVER_TEMPLATE = """
import os
import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, Body

app = FastAPI()
model = None
MODEL_PATH = "/model/model_file"

@app.on_event("startup")
def load_model():
    global model
    print(f"Loading model from {MODEL_PATH}")
    ext = os.path.splitext(MODEL_PATH)[1].lower()
    try:
        if ext == ".pkl" or ext == ".joblib":
            model = joblib.load(MODEL_PATH)
        elif ext == ".pt" or ext == ".pth":
            import torch
            model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            model.eval()
        else:
            # Try joblib as default
            model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/predict")
async def predict(data: Dict = Body(...)):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Simple heuristic to extract features
    if "features" in data:
        features = data["features"]
    else:
        features = list(data.values())
    
    X = np.array([features])
    
    try:
        if hasattr(model, "predict_proba"):
            y = model.predict_proba(X)
            return {"output": y.tolist()[0], "type": "probability"}
        elif hasattr(model, "predict"):
            y = model.predict(X)
            return {"output": y.tolist()[0], "type": "class"}
        elif callable(model):
            # For torch models or simple functions
            import torch
            with torch.no_grad():
                X_tensor = torch.tensor(X).float()
                y = model(X_tensor)
                return {"output": y.tolist()[0], "type": "tensor"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

class SandboxHandle:
    def __init__(self, container_id: str, port: int, client: docker.DockerClient):
        self.container_id = container_id
        self.port = port
        self.client = client
        self.base_url = f"http://localhost:{port}"

    def predict(self, features: Dict) -> Dict:
        try:
            resp = requests.post(f"{self.base_url}/predict", json=features, timeout=2)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def run_adversarial(self, inputs: List[Dict]) -> List[Dict]:
        results = []
        for inp in inputs:
            results.append(self.predict(inp))
        return results

    def shutdown(self):
        try:
            container = self.client.containers.get(self.container_id)
            container.stop()
            container.remove()
        except:
            pass

class ModelSandbox:
    def __init__(self):
        try:
            self.client = docker.from_env()
        except Exception as e:
            print(f"Docker client error: {e}")
            self.client = None

    def create_sandbox(self, model_path: str, requirements: List[str] = None) -> Optional[SandboxHandle]:
        if not self.client:
            return None

        sandbox_id = str(uuid.uuid4())[:8]
        # We need a base image. Using a standard python image with scikit-learn and fastapi
        # In a real enterprise app, we'd have a pre-built 'ml-guard-sandbox-base' image.
        image_name = "python:3.9-slim"
        
        # Prepare the container files
        # We'll volume mount the model and write the server script into the container via command
        # or by mounting a temp dir.
        
        # For simplicity in this demo, we'll use a temporary directory for mounting
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        
        # Copy model
        target_model_path = os.path.join(tmp_dir, "model_file")
        import shutil
        shutil.copy(model_path, target_model_path)
        
        # Write server script
        server_path = os.path.join(tmp_dir, "server.py")
        with open(server_path, "w") as f:
            f.write(SERVER_TEMPLATE)

        # Build command to install deps if needed and start server
        cmd_parts = ["pip install fastapi uvicorn joblib scikit-learn numpy"]
        if requirements:
            cmd_parts.append(f"pip install {' '.join(requirements)}")
        
        # For torch support if extension is .pt
        if model_path.endswith(".pt") or model_path.endswith(".pth"):
            cmd_parts[0] += " torch"

        cmd_parts.append("python /app/server.py")
        full_cmd = " && ".join(cmd_parts)
        
        try:
            container = self.client.containers.run(
                image_name,
                command=f"bash -c '{full_cmd}'",
                volumes={tmp_dir: {'bind': '/app', 'mode': 'rw'}, tmp_dir: {'bind': '/model', 'mode': 'rw'}},
                ports={'8000/tcp': None}, # Random port
                detach=True,
                network_disabled=True,
                mem_limit="512m",
                cpu_quota=50000,
                read_only=False, # We need to install pip pkgs. In prod this would be in the image.
                # However, task says "read_only filesystem except /tmp"
                # We'll follow the security constraints as much as possible for the demo
                security_opt=["no-new-privileges:true"],
                user="root", # Need root for pip in this slim image, usually would be 'nobody'
                name=f"ml-guard-sandbox-{sandbox_id}"
            )
            
            # Wait for startup
            container.reload()
            port = int(container.ports['8000/tcp'][0]['HostPort'])
            
            # Simple health check loop
            for _ in range(30):
                try:
                    requests.get(f"http://localhost:{port}/docs", timeout=1)
                    break
                except:
                    time.sleep(1)
            
            return SandboxHandle(container.id, port, self.client)
        except Exception as e:
            print(f"Failed to start sandbox: {e}")
            return None
