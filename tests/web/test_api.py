"""
Tests for the FastAPI web API endpoints.

Covers text management, analysis pipeline, visualization data retrieval,
and error handling for all REST routes.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from philosophical_analysis.web.app import app
from philosophical_analysis.web.routes import api as api_module


@pytest.fixture(autouse=True)
def reset_state():
    """Reset in-memory state before each test."""
    api_module._state["texts"] = {}
    api_module._state["results"] = None
    api_module._state["viz_data"] = None
    yield
    api_module._state["texts"] = {}
    api_module._state["results"] = None
    api_module._state["viz_data"] = None


@pytest.fixture
def client():
    return TestClient(app)


# -------------------------------------------------------------------------
# Health
# -------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# -------------------------------------------------------------------------
# Text management
# -------------------------------------------------------------------------

class TestTextManagement:
    def test_get_texts_empty(self, client):
        resp = client.get("/api/texts")
        assert resp.status_code == 200
        assert resp.json() == {"texts": {}}

    def test_add_text(self, client):
        resp = client.post("/api/texts", json={
            "name": "test_text",
            "content": "Some philosophical content."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "test_text" in data["texts"]
        assert data["texts"]["test_text"] == "Some philosophical content."

    def test_add_text_missing_name(self, client):
        resp = client.post("/api/texts", json={
            "name": "",
            "content": "Some content."
        })
        assert resp.status_code == 400

    def test_add_text_missing_content(self, client):
        resp = client.post("/api/texts", json={
            "name": "test",
            "content": ""
        })
        assert resp.status_code == 400

    def test_add_multiple_texts(self, client):
        client.post("/api/texts", json={"name": "a", "content": "Alpha text."})
        resp = client.post("/api/texts", json={"name": "b", "content": "Beta text."})
        texts = resp.json()["texts"]
        assert len(texts) == 2
        assert "a" in texts
        assert "b" in texts

    def test_remove_text(self, client):
        client.post("/api/texts", json={"name": "x", "content": "Delete me."})
        resp = client.delete("/api/texts/x")
        assert resp.status_code == 200
        assert "x" not in resp.json()["texts"]

    def test_remove_nonexistent_text(self, client):
        resp = client.delete("/api/texts/does_not_exist")
        assert resp.status_code == 200
        assert resp.json() == {"texts": {}}

    def test_get_texts_after_add(self, client):
        client.post("/api/texts", json={"name": "t1", "content": "Text one."})
        resp = client.get("/api/texts")
        assert "t1" in resp.json()["texts"]


# -------------------------------------------------------------------------
# File upload
# -------------------------------------------------------------------------

class TestFileUpload:
    def test_upload_single_file(self, client):
        resp = client.post("/api/upload", files=[
            ("files", ("kant.txt", b"Pure reason text.", "text/plain"))
        ])
        assert resp.status_code == 200
        assert "kant" in resp.json()["texts"]

    def test_upload_multiple_files(self, client):
        resp = client.post("/api/upload", files=[
            ("files", ("a.txt", b"Text A.", "text/plain")),
            ("files", ("b.txt", b"Text B.", "text/plain")),
        ])
        assert resp.status_code == 200
        texts = resp.json()["texts"]
        assert "a" in texts
        assert "b" in texts

    def test_upload_strips_extension(self, client):
        resp = client.post("/api/upload", files=[
            ("files", ("my_text.txt", b"Content.", "text/plain"))
        ])
        assert "my_text" in resp.json()["texts"]

    def test_upload_non_utf8_file(self, client):
        resp = client.post("/api/upload", files=[
            ("files", ("bad.txt", b"\xff\xfe", "text/plain"))
        ])
        assert resp.status_code == 400


# -------------------------------------------------------------------------
# Sample texts
# -------------------------------------------------------------------------

class TestSamples:
    def test_load_samples(self, client):
        resp = client.post("/api/samples")
        assert resp.status_code == 200
        texts = resp.json()["texts"]
        assert "kant_sample" in texts
        assert "nietzsche_sample" in texts
        assert "hume_sample" in texts
        assert len(texts) == 3

    def test_load_samples_replaces_existing(self, client):
        client.post("/api/texts", json={"name": "old", "content": "Old text."})
        resp = client.post("/api/samples")
        texts = resp.json()["texts"]
        assert "old" not in texts
        assert len(texts) == 3


# -------------------------------------------------------------------------
# Analysis
# -------------------------------------------------------------------------

class TestAnalysis:
    def test_analyze_no_texts(self, client):
        resp = client.post("/api/analyze", json={})
        assert resp.status_code == 400
        assert "No texts" in resp.json()["detail"]

    @pytest.mark.integration
    @pytest.mark.usefixtures("mock_nltk_data")
    def test_analyze_with_samples(self, client):
        client.post("/api/samples")
        resp = client.post("/api/analyze", json={
            "lsa_components": 5,
            "coherence_window": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "viz_data" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 3

    @pytest.mark.integration
    @pytest.mark.usefixtures("mock_nltk_data")
    def test_analyze_stores_state(self, client):
        client.post("/api/samples")
        client.post("/api/analyze", json={"lsa_components": 5, "coherence_window": 3})
        assert api_module._state["results"] is not None
        assert api_module._state["viz_data"] is not None

    @pytest.mark.integration
    @pytest.mark.usefixtures("mock_nltk_data")
    def test_analyze_default_params(self, client):
        client.post("/api/samples")
        resp = client.post("/api/analyze", json={})
        assert resp.status_code == 200


# -------------------------------------------------------------------------
# Visualization data endpoints
# -------------------------------------------------------------------------

class TestVizEndpoints:
    def test_results_404_before_analysis(self, client):
        resp = client.get("/api/results")
        assert resp.status_code == 404

    def test_dashboard_404_before_analysis(self, client):
        resp = client.get("/api/viz/dashboard")
        assert resp.status_code == 404

    def test_temporal_404_before_analysis(self, client):
        resp = client.get("/api/viz/temporal")
        assert resp.status_code == 404

    def test_network_404_before_analysis(self, client):
        resp = client.get("/api/viz/network")
        assert resp.status_code == 404

    def test_all_viz_404_before_analysis(self, client):
        resp = client.get("/api/viz/all")
        assert resp.status_code == 404

    @pytest.mark.integration
    @pytest.mark.usefixtures("mock_nltk_data")
    def test_dashboard_after_analysis(self, client):
        client.post("/api/samples")
        client.post("/api/analyze", json={"lsa_components": 5, "coherence_window": 3})
        resp = client.get("/api/viz/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "philosophers" in data
        assert "stats" in data

    @pytest.mark.integration
    @pytest.mark.usefixtures("mock_nltk_data")
    def test_temporal_after_analysis(self, client):
        client.post("/api/samples")
        client.post("/api/analyze", json={"lsa_components": 5, "coherence_window": 3})
        resp = client.get("/api/viz/temporal")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    @pytest.mark.integration
    @pytest.mark.usefixtures("mock_nltk_data")
    def test_network_after_analysis(self, client):
        client.post("/api/samples")
        client.post("/api/analyze", json={"lsa_components": 5, "coherence_window": 3})
        resp = client.get("/api/viz/network")
        assert resp.status_code == 200

    @pytest.mark.integration
    @pytest.mark.usefixtures("mock_nltk_data")
    def test_all_viz_after_analysis(self, client):
        client.post("/api/samples")
        client.post("/api/analyze", json={"lsa_components": 5, "coherence_window": 3})
        resp = client.get("/api/viz/all")
        assert resp.status_code == 200
        data = resp.json()
        assert "dashboard" in data
        assert "temporal" in data


# -------------------------------------------------------------------------
# SPA routing
# -------------------------------------------------------------------------

class TestSPARouting:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Philosophical Text Analysis" in resp.text

    def test_catchall_returns_html(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_catchall_deep_path(self, client):
        resp = client.get("/some/deep/path")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_static_path_not_caught(self, client):
        resp = client.get("/static/nonexistent.js")
        assert resp.status_code == 404

    def test_api_path_not_caught(self, client):
        resp = client.get("/api/nonexistent")
        assert resp.status_code in (404, 405)
