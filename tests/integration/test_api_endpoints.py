"""
Integration tests for the FastAPI application.
Tests the full request-response cycle using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    TestClient that persists for all tests in this module.
    scope=module means the app starts once — much faster.
    """
    with TestClient(app) as c:
        yield c


# ── Health Endpoint Tests ─────────────────────────────────────────────────────

class TestHealthEndpoints:

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_service_name(self, client):
        data = client.get("/").json()
        assert data["service"] == "AIOps Assistant"

    def test_liveness_returns_200(self, client):
        response = client.get("/health/live")
        assert response.status_code == 200

    def test_liveness_returns_alive_status(self, client):
        data = client.get("/health/live").json()
        assert data["status"] == "alive"
        assert "uptime_seconds" in data

    def test_readiness_returns_200(self, client):
        response = client.get("/health/ready")
        assert response.status_code == 200

    def test_readiness_returns_ready_status(self, client):
        data = client.get("/health/ready").json()
        assert data["status"] == "ready"


# ── Query Endpoint Tests ──────────────────────────────────────────────────────

class TestQueryEndpoint:

    def test_valid_query_returns_200(self, client):
        response = client.post(
            "/query",
            json={"query": "show me cpu usage"}
        )
        assert response.status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.post(
            "/query",
            json={"query": "restart the crashed pod"}
        ).json()
        assert "intent"           in data
        assert "confidence"       in data
        assert "query"            in data
        assert "suggested_action" in data

    def test_returned_intent_is_valid(self, client):
        valid_intents = {
            "metrics", "remediation", "logs", "health", "unknown"
        }
        data = client.post(
            "/query",
            json={"query": "check memory usage"}
        ).json()
        assert data["intent"] in valid_intents

    def test_confidence_between_0_and_1(self, client):
        data = client.post(
            "/query",
            json={"query": "check disk space"}
        ).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_response_echoes_original_query(self, client):
        original = "is the nginx deployment healthy"
        data     = client.post(
            "/query", json={"query": original}
        ).json()
        assert data["query"] == original

    def test_empty_query_returns_400(self, client):
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 400

    def test_missing_query_field_returns_422(self, client):
        response = client.post("/query", json={"context": "some context"})
        assert response.status_code == 422

    def test_query_with_optional_context(self, client):
        response = client.post(
            "/query",
            json={
                "query":   "check pod status",
                "context": "production namespace"
            }
        )
        assert response.status_code == 200


# ── Log Analysis Endpoint Tests ───────────────────────────────────────────────

class TestLogAnalysisEndpoint:

    def test_valid_request_returns_200(self, client):
        response = client.post(
            "/analyze/logs",
            json={"logs": ["ERROR: database failed", "INFO: retrying"]}
        )
        assert response.status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.post(
            "/analyze/logs",
            json={"logs": ["ERROR: something broke"]}
        ).json()
        for field in [
            "total_lines", "error_count",
            "has_critical", "severity_breakdown", "top_errors"
        ]:
            assert field in data, f"Missing field: {field}"

    def test_counts_are_correct(self, client):
        logs = [
            "ERROR: error one",
            "ERROR: error two",
            "INFO: all good",
            "WARNING: watch out",
        ]
        data = client.post(
            "/analyze/logs", json={"logs": logs}
        ).json()
        assert data["total_lines"] == 4
        assert data["error_count"] == 2

    def test_empty_logs_returns_400(self, client):
        response = client.post("/analyze/logs", json={"logs": []})
        assert response.status_code == 400

    def test_detects_critical(self, client):
        data = client.post(
            "/analyze/logs",
            json={"logs": ["CRITICAL: system failure", "ERROR: db down"]}
        ).json()
        assert data["has_critical"] is True


# ── Error Handling Tests ──────────────────────────────────────────────────────

class TestErrorHandling:

    def test_unknown_route_returns_404(self, client):
        response = client.get("/does-not-exist")
        assert response.status_code == 404

    def test_get_on_post_endpoint_returns_405(self, client):
        response = client.get("/query")
        assert response.status_code == 405
