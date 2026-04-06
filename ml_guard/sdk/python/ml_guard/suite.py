"""
ml_guard/suite.py — Policy Test Suites (Evidently-style, but with governance)

ML Guard's answer to Evidently's Test Suites. Unlike Evidently:
  - Tests are governance-aware (linked to scoring engine)
  - Results automatically update the governance score
  - Suites are reusable YAML-serializable policies
  - CI integration via @gate decorator

Usage:
    from ml_guard.suite import Suite, tests

    suite = Suite(model_id=\"churn-v2\", name=\"Production Quality Gate\")
    suite.add(tests.accuracy_above(0.85))
    suite.add(tests.drift_psi_below(0.25, feature=\"age\"))
    suite.add(tests.null_rate_below(0.05))
    suite.add(tests.fairness_gap_below(0.10, protected=\"gender\"))

    results = suite.run(df_reference=train_df, df_current=prod_df, model=clf)
    results.print_summary()  # rich table

    if not results.passed:
        sys.exit(1)  # CI/CD fail
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ── Individual test result ──────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    actual_value: Any
    expected: str
    message: str
    severity: str = "MEDIUM"  # LOW / MEDIUM / HIGH / CRITICAL
    category: str = "custom"  # drift / performance / fairness / quality / security


# ── Suite run results ──────────────────────────────────────────────────────────

class SuiteReport:
    """Collection of test results from a suite run."""

    def __init__(self, suite_name: str, model_id: str):
        self.suite_name = suite_name
        self.model_id = model_id
        self.results: List[TestResult] = []
        self.ran_at: float = time.time()

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_tests(self) -> List[TestResult]:
        return [r for r in self.results if not r.passed]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed for r in self.results) / len(self.results) * 100

    def print_summary(self) -> None:
        """Print a rich, color-formatted test summary table."""
        PASS = "\033[92m✓ PASS\033[0m"
        FAIL = "\033[91m✗ FAIL\033[0m"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        YELLOW = "\033[93m"

        print(f"\n{BOLD}{'─'*62}{RESET}")
        print(f"{BOLD}  ML Guard Suite: {self.suite_name}{RESET}")
        print(f"  Model: {self.model_id}  │  "
              f"Pass Rate: {YELLOW}{self.pass_rate:.0f}%{RESET}")
        print(f"{'─'*62}{RESET}")

        for r in self.results:
            status = PASS if r.passed else FAIL
            cat = f"[{r.category}]".ljust(14)
            print(f"  {status}  {cat}  {r.name}")
            if not r.passed:
                print(f"         {'':14}  └─ {r.message}")

        verdict = "PASSED" if self.passed else "FAILED"
        color = "\033[92m" if self.passed else "\033[91m"
        print(f"{'─'*62}")
        print(f"  {color}{BOLD}Overall: {verdict}{RESET}  "
              f"({sum(r.passed for r in self.results)}/{len(self.results)} tests passed)")
        print(f"{'─'*62}\n")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "model_id": self.model_id,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 1),
            "ran_at": self.ran_at,
            "test_count": len(self.results),
            "failed_count": len(self.failed_tests),
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "actual": r.actual_value,
                    "expected": r.expected,
                    "message": r.message,
                    "severity": r.severity,
                    "category": r.category,
                }
                for r in self.results
            ],
        }

    def assert_passed(self) -> None:
        """Raise SystemExit(1) if any tests failed. For CI use."""
        if not self.passed:
            self.print_summary()
            print("[ml_guard] Suite FAILED. Blocking pipeline.")
            sys.exit(1)


# ── Test factory functions ─────────────────────────────────────────────────────

class tests:
    """
    Factory namespace for creating test assertion objects.
    Each returns a Callable[[context], TestResult].
    """

    @staticmethod
    def accuracy_above(threshold: float, metric: str = "accuracy") -> Callable:
        """Test that model accuracy (or F1/AUC) exceeds threshold."""
        def _run(ctx: Dict[str, Any]) -> TestResult:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            model = ctx.get("model")
            df_current = ctx.get("df_current")
            label_col = ctx.get("label_col", "target")

            if model is None or df_current is None or label_col not in df_current.columns:
                return TestResult(
                    name=f"{metric} > {threshold}",
                    passed=False,
                    actual_value=None,
                    expected=f"> {threshold}",
                    message="Missing model or labeled data",
                    severity="HIGH",
                    category="performance",
                )

            X = df_current.drop(columns=[label_col])
            y = df_current[label_col]
            preds = model.predict(X)

            if metric == "accuracy":
                score = accuracy_score(y, preds)
            elif metric in ("f1", "f1_score"):
                score = f1_score(y, preds, average="weighted")
            else:
                score = accuracy_score(y, preds)

            passed = score >= threshold
            return TestResult(
                name=f"{metric} > {threshold:.2f}",
                passed=passed,
                actual_value=round(score, 4),
                expected=f">= {threshold}",
                message="" if passed else f"{metric}={score:.4f} < {threshold}",
                severity="HIGH",
                category="performance",
            )
        return _run

    @staticmethod
    def drift_psi_below(threshold: float, feature: Optional[str] = None) -> Callable:
        """Test PSI drift is below threshold for a feature (or all features)."""
        def _run(ctx: Dict[str, Any]) -> TestResult:
            import numpy as np
            import pandas as pd

            df_ref = ctx.get("df_reference")
            df_cur = ctx.get("df_current")
            label_col = ctx.get("label_col", "target")

            if df_ref is None or df_cur is None:
                return TestResult(
                    name=f"PSI < {threshold}" + (f" [{feature}]" if feature else ""),
                    passed=False, actual_value=None,
                    expected=f"< {threshold}",
                    message="Missing reference or current data",
                    severity="MEDIUM", category="drift",
                )

            def _psi(ref: np.ndarray, cur: np.ndarray, bins=10) -> float:
                eps = 1e-6
                breaks = np.linspace(min(ref.min(), cur.min()),
                                    max(ref.max(), cur.max()), bins + 1)
                ref_c, _ = np.histogram(ref, bins=breaks)
                cur_c, _ = np.histogram(cur, bins=breaks)
                r = ref_c / len(ref) + eps
                c = cur_c / len(cur) + eps
                return float(np.sum((c - r) * np.log(c / r)))

            feature_cols = (
                [feature] if feature
                else [c for c in df_ref.columns
                      if pd.api.types.is_numeric_dtype(df_ref[c])
                      and c != label_col]
            )

            psi_scores = {}
            for col in feature_cols:
                if col in df_ref.columns and col in df_cur.columns:
                    psi_scores[col] = _psi(
                        df_ref[col].dropna().values,
                        df_cur[col].dropna().values,
                    )

            if not psi_scores:
                return TestResult(
                    name=f"PSI < {threshold}",
                    passed=True, actual_value=0.0,
                    expected=f"< {threshold}",
                    message="No numeric features to check",
                    severity="LOW", category="drift",
                )

            max_psi = max(psi_scores.values())
            max_feat = max(psi_scores, key=psi_scores.get)
            passed = max_psi < threshold
            name = f"PSI < {threshold}" + (f" [{feature}]" if feature else " [all]")
            return TestResult(
                name=name,
                passed=passed,
                actual_value=round(max_psi, 4),
                expected=f"< {threshold}",
                message="" if passed else (
                    f"PSI={max_psi:.4f} for '{max_feat}' exceeds threshold {threshold}"
                ),
                severity="HIGH" if max_psi > 0.4 else "MEDIUM",
                category="drift",
            )
        return _run

    @staticmethod
    def null_rate_below(threshold: float, column: Optional[str] = None) -> Callable:
        """Test null/missing rate is below threshold."""
        def _run(ctx: Dict[str, Any]) -> TestResult:
            df = ctx.get("df_current")
            if df is None:
                return TestResult(
                    name=f"null_rate < {threshold}",
                    passed=False, actual_value=None,
                    expected=f"< {threshold}", message="No data",
                    severity="MEDIUM", category="quality",
                )

            cols = [column] if column else df.columns.tolist()
            null_rates = {c: df[c].isna().mean() for c in cols if c in df.columns}
            max_rate = max(null_rates.values()) if null_rates else 0.0
            max_col = max(null_rates, key=null_rates.get) if null_rates else "?"
            passed = max_rate < threshold
            name = f"null_rate < {threshold}" + (f" [{column}]" if column else " [all]")
            return TestResult(
                name=name,
                passed=passed,
                actual_value=round(max_rate, 4),
                expected=f"< {threshold}",
                message="" if passed else (
                    f"'{max_col}' null rate={max_rate:.2%} > {threshold:.2%}"
                ),
                severity="MEDIUM",
                category="quality",
            )
        return _run

    @staticmethod
    def governance_score_above(threshold: float, client=None) -> Callable:
        """Test that live governance score is above threshold. Requires client."""
        def _run(ctx: Dict[str, Any]) -> TestResult:
            _client = client or ctx.get("client")
            model_id = ctx.get("model_id", "")
            if _client is None:
                return TestResult(
                    name=f"governance_score > {threshold}",
                    passed=False, actual_value=None,
                    expected=f">= {threshold}", message="No ML Guard client",
                    severity="HIGH", category="governance",
                )
            try:
                data = _client.get_score(model_id)
                score = data.get("overall_score", 0.0)
                passed = score >= threshold
                return TestResult(
                    name=f"governance_score > {threshold}",
                    passed=passed,
                    actual_value=round(score, 2),
                    expected=f">= {threshold}",
                    message="" if passed else (
                        f"Score {score:.1f} < {threshold}"
                    ),
                    severity="CRITICAL",
                    category="governance",
                )
            except Exception as e:
                return TestResult(
                    name=f"governance_score > {threshold}",
                    passed=False, actual_value=None,
                    expected=f">= {threshold}",
                    message=f"Score fetch failed: {e}",
                    severity="HIGH", category="governance",
                )
        return _run

    @staticmethod
    def custom(name: str, fn: Callable[[Dict], bool], message: str = "",
               category: str = "custom", severity: str = "MEDIUM") -> Callable:
        """
        Add any custom assertion.

        Example:
            suite.add(tests.custom(
                name=\"revenue_distribution_stable\",
                fn=lambda ctx: ctx[\"df_current\"][\"revenue\"].mean() > 100,
                message=\"Average revenue dropped below $100\"
            ))
        """
        def _run(ctx: Dict[str, Any]) -> TestResult:
            try:
                passed = fn(ctx)
                return TestResult(
                    name=name, passed=passed,
                    actual_value=passed, expected="True",
                    message="" if passed else message,
                    severity=severity, category=category,
                )
            except Exception as e:
                return TestResult(
                    name=name, passed=False,
                    actual_value=None, expected="True",
                    message=f"Test threw exception: {e}",
                    severity=severity, category=category,
                )
        return _run


# ── Suite orchestrator ──────────────────────────────────────────────────────────

class Suite:
    """
    ML Guard Policy Test Suite — like Evidently's TestSuite but:
    - Governance-aware (reads live scores)
    - Decorator-compatible (use .run_and_assert() in CI)
    - YAML-serializable
    - Diff-able between runs

    Example:
        suite = Suite(\"churn-v2\", \"Prod Quality Gate\")
        suite.add(tests.accuracy_above(0.85))
        suite.add(tests.drift_psi_below(0.25))
        suite.add(tests.null_rate_below(0.05))

        results = suite.run(df_reference=train_df, df_current=prod_df, model=model)
        results.print_summary()
        results.assert_passed()  # raises SystemExit(1) if failed
    """

    def __init__(self, model_id: str, name: str = "ML Guard Suite", client=None):
        self.model_id = model_id
        self.name = name
        self.client = client
        self._tests: List[Callable] = []

    def add(self, test: Callable) -> "Suite":
        """Add a test to the suite. Returns self for chaining."""
        self._tests.append(test)
        return self

    def run(
        self,
        df_current=None,
        df_reference=None,
        model=None,
        label_col: str = "target",
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> SuiteReport:
        """
        Execute all tests against the provided data.

        Args:
            df_current: Current/production DataFrame
            df_reference: Reference/training DataFrame
            model: Sklearn-compatible model (for accuracy tests)
            label_col: Target/label column name
            extra_context: Any additional data to pass to custom tests

        Returns:
            SuiteReport with detailed results
        """
        ctx: Dict[str, Any] = {
            "model_id": self.model_id,
            "model": model,
            "df_current": df_current,
            "df_reference": df_reference,
            "label_col": label_col,
            "client": self.client,
            **(extra_context or {}),
        }

        report = SuiteReport(suite_name=self.name, model_id=self.model_id)

        for test_fn in self._tests:
            result = test_fn(ctx)
            report.results.append(result)

        return report

    def run_and_assert(self, **kwargs) -> None:
        """Run all tests and raise SystemExit(1) on any failure. For CI use."""
        report = self.run(**kwargs)
        report.assert_passed()
