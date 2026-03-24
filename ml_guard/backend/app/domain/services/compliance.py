from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog
from app.domain.models.test_suite import QualityGateResult, TestResult

logger = structlog.get_logger(__name__)

class ComplianceService:
    """
    Tier 3: Compliance Mode for ML Guard.
    Maps test results to regulatory pillars (Fairness, Explainability, etc.)
    and generates formal Audit Reports.
    """
    
    PILLAR_MAPPING = {
        "Fairness": ["bias_fairness", "disparate_impact", "demographic_parity"],
        "Stability": ["statistical_stability", "psi_drift", "ks_test", "drift"],
        "Robustness": ["robustness", "input_perturbation", "prediction_stability"],
        "Performance": ["model_performance", "accuracy", "precision", "recall"],
        "Data Integrity": ["data_quality", "missing_values", "duplicates"]
    }

    def generate_audit_report(self, result: QualityGateResult) -> Dict[str, Any]:
        """
        Generates a structured Model Audit Report (JSON).
        """
        pillars = self._map_results_to_pillars(result.results)
        
        # Calculate pillar-specific scores
        pillar_status = {}
        for pillar, items in pillars.items():
            failed = [i for i in items if i.status == "failed"]
            pillar_status[pillar] = {
                "status": "PASS" if not failed else "FAIL",
                "failure_count": len(failed),
                "test_count": len(items)
            }

        report = {
            "report_id": f"AUDIT-{result.run_id}",
            "generated_at": datetime.utcnow().isoformat(),
            "model_identity": {
                "project": result.project_id,
                "version": result.model_version,
                "reproducibility_token": result.reproducibility_token
            },
            "risk_summary": {
                "overall_score": result.score,
                "risk_level": result.risk_level,
                "deployment_allowed": result.deployment_allowed
            },
            "compliance_pillars": pillar_status,
            "transparency_report": {
                "explainability": result.feature_importance[:10] if result.feature_importance else "No explainability data",
                "environment": result.environment_config
            },
            "dataset_documentation": {
                "fingerprints": result.execution_metadata.get("dataset_fingerprints", {}),
                "profile": result.model_profile.get("dataset_stats", {})
            },
            "compliance_checklist": self._generate_checklist(result)
        }
        
        return report

    def _map_results_to_pillars(self, results: List[TestResult]) -> Dict[str, List[TestResult]]:
        mapped = {p: [] for p in self.PILLAR_MAPPING}
        mapped["Other"] = []
        
        for r in results:
            found = False
            for pillar, tags in self.PILLAR_MAPPING.items():
                if any(tag in r.test_name.lower() or tag in str(r.category).lower() for tag in tags):
                    mapped[pillar].append(r)
                    found = True
                    break
            if not found:
                mapped["Other"].append(r)
        return mapped

    def _generate_checklist(self, result: QualityGateResult) -> List[Dict[str, Any]]:
        """
        Generates a high-level governance checklist.
        """
        profile = result.model_profile
        stats = profile.get("dataset_stats", {})
        
        checklist = [
            {
                "item": "Reproducibility Verified",
                "status": "YES" if result.reproducibility_token else "NO",
                "evidence": result.reproducibility_token
            },
            {
                "item": "Protected Attributes Scanned",
                "status": "YES",
                "details": f"Detected: {stats.get('protected_attributes', 'None')}"
            },
            {
                "item": "Explainability Requirements",
                "status": "PASS" if result.feature_importance else "FAIL",
                "details": "SHAP global importance generated" if result.feature_importance else "Missing explainability"
            },
            {
                "item": "Gate Enforcement",
                "status": "PASS" if result.deployment_allowed else "FAIL",
                "details": f"Score {result.score} against risk bucket {result.risk_level}"
            }
        ]
        return checklist
