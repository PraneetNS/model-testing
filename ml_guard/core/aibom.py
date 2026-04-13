import hashlib
import json
import requests
import importlib.metadata
import datetime
from typing import List, Dict, Any

def compute_sha256(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception:
        return ""

def get_installed_dependencies() -> List[Dict[str, str]]:
    deps = []
    try:
        for dist in importlib.metadata.distributions():
            deps.append({
                "name": dist.metadata["Name"],
                "version": dist.version,
                "sha256_hash": "", # Python doesn't make this easy to get for installed packages
                "source": "pypi"
            })
    except Exception:
        pass
    return deps

def check_osv_vulnerabilities(package_name: str, version: str) -> List[Dict[str, Any]]:
    vulnerabilities = []
    try:
        url = "https://api.osv.dev/v1/query"
        payload = {
            "version": version,
            "package": {"name": package_name, "ecosystem": "PyPI"}
        }
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "vulns" in data:
                for vuln in data["vulns"]:
                    vulnerabilities.append({
                        "package": package_name,
                        "version": version,
                        "cve_id": vuln.get("id"),
                        "severity": vuln.get("database_specific", {}).get("severity", "UNKNOWN")
                    })
    except Exception:
        pass
    return vulnerabilities

def fetch_hf_metadata(hf_model_id: str) -> Dict[str, Any]:
    try:
        url = f"https://huggingface.co/api/models/{hf_model_id}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "license": data.get("cardData", {}).get("license") or data.get("license"),
                "tags": data.get("tags", []),
                "pipeline_tag": data.get("pipeline_tag")
            }
    except Exception:
        pass
    return {}

def generate_aibom(model_path: str, dataset_paths: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Function generate_aibom(model_path, dataset_paths, metadata) that:
    a. Computes SHA-256 of the model file and each dataset file.
    b. Extracts Python dependency versions from the current venv using importlib.metadata.
    c. If hf_model_id is provided, fetches the HuggingFace model card metadata.
    d. Checks each dependency name+version against the OSV.dev API for known CVEs.
    e. Computes a final aibom_hash over the serialized JSON.
    """
    # a. Compute hashes
    model_hash = compute_sha256(model_path)
    training_datasets = []
    for dp in dataset_paths:
        if dp and isinstance(dp, str):
            d_hash = compute_sha256(dp)
            name = dp.replace("\\", "/").split("/")[-1]
            training_datasets.append({
                "name": name,
                "source": "local",
                "sha256_hash": d_hash,
                "record_count": metadata.get("record_counts", {}).get(name, 0),
                "date_collected": datetime.datetime.utcnow().isoformat(),
                "license": metadata.get("dataset_license", "unknown"),
                "known_poisoning_cves": []
            })

    # b. Dependencies
    dependencies = get_installed_dependencies()
    
    # c. HF Metadata
    hf_id = metadata.get("hf_model_id")
    hf_meta = fetch_hf_metadata(hf_id) if hf_id else {}
    
    # d. CVE Checks
    cve_alerts = []
    for dep in dependencies:
        vulns = check_osv_vulnerabilities(dep["name"], dep["version"])
        cve_alerts.extend(vulns)

    aibom = {
        "model_id": metadata.get("model_id"),
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "schema_version": "1.0",
        "base_model": {
            "name": metadata.get("model_name", "unknown"),
            "source_url": metadata.get("source_url"),
            "sha256_hash": model_hash,
            "hf_model_id": hf_id,
            "license": hf_meta.get("license") or metadata.get("license", "unknown")
        },
        "training_datasets": training_datasets,
        "dependencies": dependencies,
        "training_framework": {
            "name": metadata.get("framework", "unknown"),
            "version": metadata.get("framework_version", "unknown")
        },
        "cve_alerts": cve_alerts
    }
    
    # e. Compute final hash
    # We remove cve_alerts from hash calculation or keep it? 
    # Usually AIBOM hash is over the manifest content.
    aibom_json = json.dumps(aibom, sort_keys=True)
    aibom["aibom_hash"] = hashlib.sha256(aibom_json.encode()).hexdigest()
    
    return aibom
