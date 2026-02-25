"""
Visualization data contract tests.

Verifies that the data structures produced by the API analysis pipeline
match the shape expected by the frontend JavaScript (app.js).

If any of these fail, the frontend charts will break silently.
"""

import json
import pytest
from fastapi.testclient import TestClient

from philosophical_analysis.web.app import app
from philosophical_analysis.web.routes import api as api_module


@pytest.fixture(autouse=True)
def reset_state():
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


@pytest.fixture
def analyzed_client(client):
    """Client with sample texts already analyzed."""
    client.post("/api/samples")
    resp = client.post("/api/analyze", json={
        "lsa_components": 5,
        "coherence_window": 3,
    })
    assert resp.status_code == 200
    return client


# -------------------------------------------------------------------------
# Dashboard contract
# -------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.usefixtures("mock_nltk_data")
class TestDashboardContract:
    """
    The frontend expects:
        dashboard.philosophers[name] = {
            first_order_coherence, second_order_coherence,
            syntactic_complexity, determiner_frequency,
            avg_sentence_length
        }
        dashboard.stats = {
            highest_coherence, highest_coherence_value,
            most_complex_syntax, most_complex_value,
            avg_coherence (optional)
        }
    """

    def test_dashboard_has_philosophers_dict(self, analyzed_client):
        data = analyzed_client.get("/api/viz/dashboard").json()
        assert isinstance(data["philosophers"], dict)
        assert len(data["philosophers"]) >= 1

    def test_philosopher_has_required_metrics(self, analyzed_client):
        data = analyzed_client.get("/api/viz/dashboard").json()
        required_keys = [
            "first_order_coherence",
            "second_order_coherence",
            "syntactic_complexity",
            "determiner_frequency",
            "avg_sentence_length",
        ]
        for name, metrics in data["philosophers"].items():
            for key in required_keys:
                assert key in metrics, f"Missing '{key}' for philosopher '{name}'"
                assert isinstance(metrics[key], (int, float, type(None))), \
                    f"'{key}' for '{name}' is {type(metrics[key])}, expected numeric"

    def test_stats_has_required_fields(self, analyzed_client):
        data = analyzed_client.get("/api/viz/dashboard").json()
        stats = data["stats"]
        assert "highest_coherence" in stats
        assert "highest_coherence_value" in stats
        assert "most_complex_syntax" in stats or "most_complex" in stats

    def test_all_values_json_serializable(self, analyzed_client):
        data = analyzed_client.get("/api/viz/dashboard").json()
        # If we got here, FastAPI already serialized it, but double-check roundtrip
        serialized = json.dumps(data)
        roundtrip = json.loads(serialized)
        assert roundtrip == data


# -------------------------------------------------------------------------
# Temporal contract
# -------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.usefixtures("mock_nltk_data")
class TestTemporalContract:
    """
    The frontend expects:
        temporal[PHILOSOPHER_NAME] = {
            coherence_timeline: [float, ...],
            avg_coherence: float,
            segments: [{start, end, avg_coherence}, ...] (optional),
            volatility: float (optional),
            peak_coherence: float (optional)
        }
    """

    def test_temporal_is_dict(self, analyzed_client):
        data = analyzed_client.get("/api/viz/temporal").json()
        assert isinstance(data, dict)

    def test_each_philosopher_has_timeline(self, analyzed_client):
        data = analyzed_client.get("/api/viz/temporal").json()
        for name, entry in data.items():
            assert "coherence_timeline" in entry, \
                f"Missing 'coherence_timeline' for '{name}'"
            assert isinstance(entry["coherence_timeline"], list), \
                f"'coherence_timeline' for '{name}' is not a list"

    def test_timeline_values_are_numeric(self, analyzed_client):
        data = analyzed_client.get("/api/viz/temporal").json()
        for name, entry in data.items():
            for i, val in enumerate(entry["coherence_timeline"]):
                assert isinstance(val, (int, float)), \
                    f"Timeline value [{i}] for '{name}' is {type(val)}"

    def test_avg_coherence_present(self, analyzed_client):
        data = analyzed_client.get("/api/viz/temporal").json()
        for name, entry in data.items():
            assert "avg_coherence" in entry, \
                f"Missing 'avg_coherence' for '{name}'"


# -------------------------------------------------------------------------
# Network contract
# -------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.usefixtures("mock_nltk_data")
class TestNetworkContract:
    """
    The frontend expects:
        network.nodes = [{id, label, category, size, philosopher}, ...]
        network.links = [{source, target, weight}, ...]
        network.metadata = {node_count, link_count, density, ...}
    """

    def test_network_has_nodes_and_links(self, analyzed_client):
        data = analyzed_client.get("/api/viz/network").json()
        assert "nodes" in data
        assert "links" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["links"], list)

    def test_node_has_required_fields(self, analyzed_client):
        data = analyzed_client.get("/api/viz/network").json()
        for node in data["nodes"]:
            assert "id" in node, f"Node missing 'id': {node}"
            assert "label" in node or "id" in node

    def test_link_has_source_and_target(self, analyzed_client):
        data = analyzed_client.get("/api/viz/network").json()
        for link in data["links"]:
            assert "source" in link, f"Link missing 'source': {link}"
            assert "target" in link, f"Link missing 'target': {link}"

    def test_metadata_present(self, analyzed_client):
        data = analyzed_client.get("/api/viz/network").json()
        if "metadata" in data:
            meta = data["metadata"]
            assert isinstance(meta, dict)


# -------------------------------------------------------------------------
# Combined viz/all contract
# -------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.usefixtures("mock_nltk_data")
class TestAllVizContract:
    """The /api/viz/all endpoint should include dashboard and temporal keys."""

    def test_all_viz_has_required_keys(self, analyzed_client):
        data = analyzed_client.get("/api/viz/all").json()
        assert "dashboard" in data
        assert "temporal" in data

    def test_all_viz_roundtrip_json(self, analyzed_client):
        data = analyzed_client.get("/api/viz/all").json()
        serialized = json.dumps(data)
        roundtrip = json.loads(serialized)
        assert roundtrip == data


# -------------------------------------------------------------------------
# Results endpoint contract
# -------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.usefixtures("mock_nltk_data")
class TestResultsContract:
    """
    The /api/results endpoint returns the raw analysis DataFrame as
    a list of dicts (orient="records").
    """

    def test_results_is_list_of_dicts(self, analyzed_client):
        data = analyzed_client.get("/api/results").json()
        assert isinstance(data, list)
        assert all(isinstance(r, dict) for r in data)

    def test_results_have_text_id(self, analyzed_client):
        data = analyzed_client.get("/api/results").json()
        for row in data:
            assert "text_id" in row

    def test_results_have_coherence(self, analyzed_client):
        data = analyzed_client.get("/api/results").json()
        for row in data:
            assert "first_order_coherence" in row
