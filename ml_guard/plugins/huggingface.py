"""
huggingface.py — HuggingFace Hub Integration Plugin for ML Guard

Enables zero-upload governance by pulling models and datasets directly
from HuggingFace Hub (700k+ models, 200k+ datasets).

Security:
  - HF tokens are NEVER persisted — used only for the duration of the request.
  - All repo IDs are validated against a strict regex before any API call.
  - Downloaded models must be sandboxed before inference.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Validation ────────────────────────────────────────────────────────────────

REPO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+$")

PERMISSIVE_LICENSES = frozenset([
    "apache-2.0", "mit", "cc-by-4.0", "cc0-1.0", "cc0",
    "openrail", "openrail++", "bigscience-openrail-m",
    "bsd-3-clause", "bsd-2-clause", "unlicense",
])

# Well-known model file names in priority order
MODEL_FILE_PRIORITIES = [
    "pytorch_model.bin",
    "model.safetensors",
    "model.pkl",
    "model.joblib",
    "model.onnx",
    "tf_model.h5",
    "flax_model.msgpack",
]

HF_DATASET_CACHE = Path(tempfile.gettempdir()) / "mlguard_hf_datasets"


def _validate_repo_id(repo_id: str) -> None:
    """Raise ValueError if repo_id doesn't match the HuggingFace pattern."""
    if not REPO_ID_PATTERN.match(repo_id):
        raise ValueError(
            f"Invalid repo_id '{repo_id}'. "
            "Must match pattern: <namespace>/<repo-name> "
            "(alphanumeric, hyphens, underscores, dots only)."
        )


def _sha256_file(path: str) -> str:
    """Compute SHA-256 of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Plugin Class ──────────────────────────────────────────────────────────────

class HuggingFacePlugin:
    """
    Pulls models and datasets from HuggingFace Hub for ML Guard governance.

    Parameters
    ----------
    hf_token : str | None
        Optional HF access token for gated models / private repos.
        Never persisted — discarded after use.
    """

    def __init__(self, hf_token: Optional[str] = None):
        self._token = hf_token or None  # Normalize empty string to None

    # ── Model Pull ────────────────────────────────────────────────────────────

    def pull_model(
        self,
        repo_id: str,
        revision: str = "main",
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Download a model file from HuggingFace Hub.

        If *filename* is None, auto-detects by scanning the repo's file list
        for well-known model filenames.

        Returns
        -------
        dict with keys:
            local_path, sha256, repo_id, revision,
            model_card_url, license, pipeline_tag, downloads_last_month
        """
        _validate_repo_id(repo_id)

        from huggingface_hub import hf_hub_download, HfApi

        api = HfApi(token=self._token)

        # Auto-detect filename if not provided
        if filename is None:
            repo_files = api.list_repo_files(repo_id, revision=revision)
            filename = self._detect_model_file(repo_files)
            if filename is None:
                raise FileNotFoundError(
                    f"No recognized model file found in {repo_id}. "
                    f"Files: {repo_files[:20]}. "
                    f"Specify 'filename' explicitly."
                )

        logger.info("hf_pull_model repo=%s file=%s rev=%s", repo_id, filename, revision)

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=self._token,
        )

        sha256 = _sha256_file(local_path)

        # Fetch repo metadata
        try:
            info = api.model_info(repo_id, revision=revision)
            license_id = getattr(info, "card_data", None)
            if license_id and hasattr(license_id, "license"):
                license_id = license_id.license
            else:
                license_id = getattr(info, "license", None)
            pipeline_tag = getattr(info, "pipeline_tag", None)
            downloads = getattr(info, "downloads", 0)
        except Exception:
            license_id = None
            pipeline_tag = None
            downloads = 0

        return {
            "local_path": str(local_path),
            "sha256": sha256,
            "repo_id": repo_id,
            "revision": revision,
            "filename": filename,
            "model_card_url": f"https://huggingface.co/{repo_id}",
            "license": license_id,
            "pipeline_tag": pipeline_tag,
            "downloads_last_month": downloads,
        }

    # ── Dataset Pull ──────────────────────────────────────────────────────────

    def pull_dataset(
        self,
        repo_id: str,
        split: str = "test",
        max_rows: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Load a dataset from HuggingFace Datasets and save as CSV.

        Returns
        -------
        dict with keys:
            local_csv_path, repo_id, split, row_count, column_names, feature_types
        """
        _validate_repo_id(repo_id)

        from datasets import load_dataset

        logger.info("hf_pull_dataset repo=%s split=%s max=%d", repo_id, split, max_rows)

        ds = load_dataset(repo_id, split=split, token=self._token)
        df = ds.to_pandas()

        # Trim to max_rows
        if len(df) > max_rows:
            df = df.head(max_rows)

        # Save to temp CSV
        HF_DATASET_CACHE.mkdir(parents=True, exist_ok=True)
        safe_name = repo_id.replace("/", "__")
        csv_path = HF_DATASET_CACHE / f"{safe_name}_{split}_{len(df)}.csv"
        df.to_csv(csv_path, index=False)

        return {
            "local_csv_path": str(csv_path),
            "repo_id": repo_id,
            "split": split,
            "row_count": len(df),
            "column_names": list(df.columns),
            "feature_types": {col: str(df[col].dtype) for col in df.columns},
        }

    # ── Model Card Risk Analysis ──────────────────────────────────────────────

    def get_model_card_risks(self, repo_id: str) -> Dict[str, Any]:
        """
        Fetch and analyze the model card for governance risk signals.

        Risk flags:
          - no_model_card:       card is missing or empty
          - no_bias_disclosure:  no bias / limitations section
          - restrictive_license: license not in permissive set
        """
        _validate_repo_id(repo_id)

        from huggingface_hub import HfApi

        api = HfApi(token=self._token)
        risk_flags: List[str] = []

        try:
            info = api.model_info(repo_id)
        except Exception as e:
            logger.warning("Failed to fetch model info for %s: %s", repo_id, e)
            return {
                "has_model_card": False,
                "license": None,
                "has_limitations_section": False,
                "has_bias_section": False,
                "risk_flags": ["no_model_card"],
            }

        # Extract card data
        card_data = getattr(info, "card_data", None)
        card_text = getattr(info, "card_text", "") or ""
        has_card = bool(card_text.strip())

        if not has_card:
            risk_flags.append("no_model_card")

        # License check
        license_id = None
        if card_data and hasattr(card_data, "license"):
            license_id = card_data.license
        if not license_id:
            license_id = getattr(info, "license", None)

        if license_id and license_id.lower() not in PERMISSIVE_LICENSES:
            risk_flags.append("restrictive_license")

        # Card section analysis
        card_lower = card_text.lower()
        has_limitations = any(
            phrase in card_lower
            for phrase in ["limitation", "known issues", "out-of-scope", "risks"]
        )
        has_bias = any(
            phrase in card_lower
            for phrase in ["bias", "fairness", "ethical", "demographic"]
        )

        if has_card and not has_bias:
            risk_flags.append("no_bias_disclosure")

        if has_card and not has_limitations:
            risk_flags.append("no_limitations_section")

        return {
            "has_model_card": has_card,
            "license": license_id,
            "has_limitations_section": has_limitations,
            "has_bias_section": has_bias,
            "risk_flags": risk_flags,
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "downloads": getattr(info, "downloads", 0),
            "likes": getattr(info, "likes", 0),
        }

    # ── Search ────────────────────────────────────────────────────────────────

    def search_models(
        self,
        query: str,
        task: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search HuggingFace Hub for models.

        Returns a list of dicts with:
            repo_id, downloads, likes, pipeline_tag, license, has_model_card
        """
        from huggingface_hub import HfApi

        api = HfApi(token=self._token)

        kwargs: Dict[str, Any] = {"search": query, "limit": limit, "sort": "downloads", "direction": -1}
        if task:
            kwargs["filter"] = task

        models = api.list_models(**kwargs)

        results = []
        for m in models:
            card_data = getattr(m, "card_data", None)
            license_id = None
            if card_data and hasattr(card_data, "license"):
                license_id = card_data.license
            results.append({
                "repo_id": m.id,
                "downloads": getattr(m, "downloads", 0),
                "likes": getattr(m, "likes", 0),
                "pipeline_tag": getattr(m, "pipeline_tag", None),
                "license": license_id,
                "has_model_card": bool(getattr(m, "card_data", None)),
            })

        return results

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_model_file(repo_files: List[str]) -> Optional[str]:
        """Pick the best model file from a list of repo files."""
        for priority_name in MODEL_FILE_PRIORITIES:
            for f in repo_files:
                if f == priority_name or f.endswith("/" + priority_name):
                    return f

        # Fallback: any .bin, .safetensors, .pkl, .onnx, .joblib
        model_extensions = {".bin", ".safetensors", ".pkl", ".joblib", ".onnx", ".h5"}
        for f in repo_files:
            ext = os.path.splitext(f)[1].lower()
            if ext in model_extensions:
                return f

        return None
