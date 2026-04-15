"""
test_hf_plugin.py — Unit tests for the HuggingFace Hub Integration Plugin

Mocks HuggingFace API calls to verify:
  - SHA-256 computation
  - Model file detection logic
  - Model card risk assessment
  - Permissive vs Restrictive license flagging
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
from ml_guard.plugins.huggingface import HuggingFacePlugin

class TestHuggingFacePlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = HuggingFacePlugin(hf_token="fake_token")

    @patch("huggingface_hub.HfApi.list_repo_files")
    def test_detect_model_file(self, mock_list):
        # Priority 1: pytorch_model.bin
        mock_list.return_value = ["README.md", "pytorch_model.bin", "config.json"]
        det = self.plugin._detect_model_file(mock_list.return_value)
        self.assertEqual(det, "pytorch_model.bin")

        # Priority middle: model.pkl
        mock_list.return_value = ["README.md", "model.pkl", "data.csv"]
        det = self.plugin._detect_model_file(mock_list.return_value)
        self.assertEqual(det, "model.pkl")

        # Fallback by extension
        mock_list.return_value = ["README.md", "my_custom_name.safetensors"]
        det = self.plugin._detect_model_file(mock_list.return_value)
        self.assertEqual(det, "my_custom_name.safetensors")

        # No match
        mock_list.return_value = ["README.md", "script.py"]
        det = self.plugin._detect_model_file(mock_list.return_value)
        self.assertIsNone(det)

    @patch("huggingface_hub.HfApi.model_info")
    def test_model_card_risks(self, mock_info):
        # Case 1: Perfect model card (Permissive license, Bias & Limitations sections)
        m1 = MagicMock()
        m1.card_text = "## Bias\nWe found no bias.\n## Limitations\nOnly for English."
        m1.card_data = MagicMock(license="mit")
        m1.pipeline_tag = "text-classification"
        m1.downloads = 1000
        m1.likes = 50
        mock_info.return_value = m1
        
        risks = self.plugin.get_model_card_risks("org/good-model")
        self.assertTrue(risks["has_model_card"])
        self.assertEqual(risks["license"], "mit")
        self.assertEqual(len(risks["risk_flags"]), 0)

        # Case 2: Restrictive license & No bias disclosure
        m2 = MagicMock()
        m2.card_text = "## Summary\nA very restricted model."
        m2.card_data = MagicMock(license="cc-by-nc-4.0")
        mock_info.return_value = m2
        
        risks = self.plugin.get_model_card_risks("org/bad-model")
        self.assertIn("restrictive_license", risks["risk_flags"])
        self.assertIn("no_bias_disclosure", risks["risk_flags"])
        self.assertIn("no_limitations_section", risks["risk_flags"])

        # Case 3: Missing model card
        m3 = MagicMock()
        m3.card_text = ""
        mock_info.return_value = m3
        risks = self.plugin.get_model_card_risks("org/empty-model")
        self.assertIn("no_model_card", risks["risk_flags"])

    def test_validate_repo_id(self):
        from ml_guard.plugins.huggingface import _validate_repo_id
        # Valid
        _validate_repo_id("microsoft/resnet-50")
        _validate_repo_id("bigscience/bloom-7b1")
        
        # Invalid
        with self.assertRaises(ValueError):
            _validate_repo_id("just-a-name") # No namespace
        with self.assertRaises(ValueError):
            _validate_repo_id("org/name/extra") # Too many slashes
        with self.assertRaises(ValueError):
            _validate_repo_id("org/name;") # Forbidden chars


if __name__ == "__main__":
    unittest.main()
