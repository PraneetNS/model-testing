"""
Model Registry - Manage ML models and their metadata
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import structlog

from ..core.config import settings

logger = structlog.get_logger(__name__)

class ModelRegistry:
    """Service for managing ML models and their metadata."""

    def __init__(self):
        self.models_dir = os.path.join(settings.STORAGE_PATH, "models")
        self.metadata_dir = os.path.join(settings.STORAGE_PATH, "metadata")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    async def register_model(
        self,
        project_id: str,
        model_version: str,
        model_path: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Register a new model version.

        Args:
            project_id: FireFlink project identifier
            model_version: Model version string
            model_path: Path to the model file
            metadata: Model metadata

        Returns:
            str: Registration ID
        """

        registration_id = f"{project_id}-{model_version}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        model_metadata = {
            "registration_id": registration_id,
            "project_id": project_id,
            "model_version": model_version,
            "model_path": model_path,
            "registered_at": datetime.utcnow().isoformat(),
            "metadata": metadata
        }

        # Save metadata
        metadata_path = os.path.join(self.metadata_dir, f"{registration_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)

        logger.info(
            "Model registered",
            registration_id=registration_id,
            project_id=project_id,
            model_version=model_version
        )

        return registration_id

    async def get_model_metadata(self, project_id: str, model_version: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific model version.

        Args:
            project_id: Project identifier
            model_version: Model version

        Returns:
            Optional[Dict]: Model metadata if found
        """

        # Find metadata file for this model version
        for filename in os.listdir(self.metadata_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.metadata_dir, filename), 'r') as f:
                        metadata = json.load(f)

                    if (metadata.get('project_id') == project_id and
                        metadata.get('model_version') == model_version):
                        return metadata
                except Exception:
                    continue

        return None

    async def list_model_versions(self, project_id: str) -> List[Dict[str, Any]]:
        """
        List all model versions for a project.

        Args:
            project_id: Project identifier

        Returns:
            List[Dict]: List of model versions
        """

        versions = []
        for filename in os.listdir(self.metadata_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.metadata_dir, filename), 'r') as f:
                        metadata = json.load(f)

                    if metadata.get('project_id') == project_id:
                        versions.append(metadata)
                except Exception:
                    continue

        # Sort by registration date (newest first)
        versions.sort(key=lambda x: x.get('registered_at', ''), reverse=True)
        return versions

    async def project_exists(self, project_id: str) -> bool:
        """
        Check if a project exists (has any registered models).

        Args:
            project_id: Project identifier

        Returns:
            bool: True if project exists
        """

        for filename in os.listdir(self.metadata_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.metadata_dir, filename), 'r') as f:
                        metadata = json.load(f)

                    if metadata.get('project_id') == project_id:
                        return True
                except Exception:
                    continue

        return False

    async def model_version_exists(self, project_id: str, model_version: str) -> bool:
        """
        Check if a specific model version exists.

        Args:
            project_id: Project identifier
            model_version: Model version

        Returns:
            bool: True if model version exists
        """

        return await self.get_model_metadata(project_id, model_version) is not None

    async def get_model_path(self, project_id: str, model_version: str) -> Optional[str]:
        """
        Get the file path for a specific model version.

        Args:
            project_id: Project identifier
            model_version: Model version

        Returns:
            Optional[str]: Model file path if found
        """

        metadata = await self.get_model_metadata(project_id, model_version)
        if metadata:
            return metadata.get('model_path')
        return None

    async def delete_model_version(self, project_id: str, model_version: str) -> bool:
        """
        Delete a model version and its metadata.

        Args:
            project_id: Project identifier
            model_version: Model version

        Returns:
            bool: True if deleted successfully
        """

        metadata = await self.get_model_metadata(project_id, model_version)
        if not metadata:
            return False

        registration_id = metadata.get('registration_id')

        # Delete metadata file
        metadata_path = os.path.join(self.metadata_dir, f"{registration_id}.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        # Delete model file if it exists
        model_path = metadata.get('model_path')
        if model_path and os.path.exists(model_path):
            os.remove(model_path)

        logger.info(
            "Model version deleted",
            project_id=project_id,
            model_version=model_version,
            registration_id=registration_id
        )

        return True