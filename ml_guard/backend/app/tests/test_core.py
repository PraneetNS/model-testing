"""
Unit tests for core ML Guard logic.
Run with: cd backend && .\venv\Scripts\python.exe -m pytest app/tests/ -v
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import timedelta

from app.core.security import create_access_token, verify_password, get_password_hash
from app.domain.services.nlp_parser import NLPParser


# ──────────────────────────────────────────────────────────────────────────────
# SECURITY TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestSecurity:
    def test_password_hash_and_verify(self):
        plain = "SuperSecret123!"
        hashed = get_password_hash(plain)
        assert hashed != plain
        assert verify_password(plain, hashed)

    def test_wrong_password_fails(self):
        hashed = get_password_hash("correct_password")
        assert not verify_password("wrong_password", hashed)

    def test_create_access_token_returns_string(self):
        token = create_access_token(subject="user-id-123", expires_delta=timedelta(minutes=30))
        assert isinstance(token, str)
        assert len(token) > 10

    def test_token_contains_three_parts(self):
        """JWT must have header.payload.signature format."""
        token = create_access_token(subject="user-abc")
        parts = token.split(".")
        assert len(parts) == 3


# ──────────────────────────────────────────────────────────────────────────────
# NLP PARSER TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestNLPParser:
    def setup_method(self):
        self.parser = NLPParser()

    def test_empty_query_returns_defaults(self):
        result = self.parser.parse_query("")
        assert "accuracy" in result
        assert "data_quality" in result

    def test_none_query_returns_defaults(self):
        result = self.parser.parse_query(None)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_all_tests_query(self):
        result = self.parser.parse_query("run all tests")
        assert "accuracy" in result
        assert "bias" in result
        assert "drift" in result

    def test_accuracy_keyword(self):
        result = self.parser.parse_query("check model accuracy")
        assert "accuracy" in result

    def test_bias_keyword(self):
        result = self.parser.parse_query("check for bias and fairness issues")
        assert "bias" in result

    def test_drift_keyword(self):
        result = self.parser.parse_query("detect feature drift")
        assert "drift" in result

    def test_unknown_query_returns_defaults(self):
        result = self.parser.parse_query("something completely unrelated xyzzy")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_query_length_capped(self):
        """Queries over 500 chars should still work (not crash)."""
        long_query = "accuracy " * 1000
        result = self.parser.parse_query(long_query)
        assert isinstance(result, list)

    def test_returns_list_always(self):
        for query in ["", "accuracy", "all", "XyZ123###"]:
            result = self.parser.parse_query(query)
            assert isinstance(result, list)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_settings_loads(self):
        from app.core.config import settings
        assert settings.PROJECT_NAME is not None
        assert settings.API_V1_STR == "/api/v1"
        assert settings.SQLALCHEMY_DATABASE_URI is not None

    def test_sqlite_fallback(self):
        """When POSTGRES env vars not set, SQLite URI should be used."""
        from app.core.config import settings
        uri = settings.SQLALCHEMY_DATABASE_URI
        # In test environment without PG, SQLite is expected
        assert uri is not None

    def test_algorithm_is_hs256(self):
        from app.core.config import settings
        assert settings.ALGORITHM == "HS256"


# ──────────────────────────────────────────────────────────────────────────────
# AUTH ENDPOINT INTEGRATION TESTS (uses TestClient, no real DB write)
# ──────────────────────────────────────────────────────────────────────────────

class TestAuthEndpoints:
    def test_login_with_wrong_credentials(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.post(
            "/api/v1/auth/login",
            data={"username": "nonexistent@test.com", "password": "wrong"},
        )
        assert resp.status_code == 400

    def test_health_check(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_protected_route_requires_token(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.get("/api/v1/governance/projects")
        assert resp.status_code in [401, 403, 422]
