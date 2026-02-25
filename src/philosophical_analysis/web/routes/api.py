"""
API routes for philosophical text analysis.

Provides REST endpoints for uploading texts, running analysis,
and retrieving visualization data.
"""

import hashlib
import json
import logging
from typing import Dict, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

from philosophical_analysis.core.integrated_analyzer import IntegratedPhilosophicalAnalyzer
from philosophical_analysis.visualization.generator import VisualizationGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# In-memory session state (single-user; mirrors former Streamlit session)
# ---------------------------------------------------------------------------

_state: Dict = {
    "texts": {},
    "results": None,
    "viz_data": None,
    "analysis_hash": None,
}


def _compute_analysis_hash(texts: Dict[str, str], params: dict) -> str:
    """Compute a deterministic hash of texts + params to detect unchanged inputs."""
    payload = json.dumps({"texts": texts, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    name: str
    content: str

class AnalysisParams(BaseModel):
    lsa_components: int = 10
    coherence_window: int = 5

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Text management
# ---------------------------------------------------------------------------

@router.get("/texts")
async def get_texts():
    return {"texts": {k: v for k, v in _state["texts"].items()}}


@router.post("/texts")
async def add_text(payload: TextInput):
    if not payload.name or not payload.content:
        raise HTTPException(400, "Name and content are required")
    _state["texts"][payload.name] = payload.content
    _state["analysis_hash"] = None  # invalidate cache
    return {"texts": {k: v for k, v in _state["texts"].items()}}


@router.delete("/texts/{name}")
async def remove_text(name: str):
    _state["texts"].pop(name, None)
    _state["analysis_hash"] = None  # invalidate cache
    return {"texts": {k: v for k, v in _state["texts"].items()}}


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    for f in files:
        content = await f.read()
        text_name = f.filename.rsplit(".", 1)[0] if f.filename else f"text_{len(_state['texts'])}"
        try:
            _state["texts"][text_name] = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, f"Could not decode {f.filename} as UTF-8")
    _state["analysis_hash"] = None  # invalidate cache
    return {"texts": {k: v for k, v in _state["texts"].items()}}


@router.post("/samples")
async def load_samples():
    _state["analysis_hash"] = None  # invalidate cache
    _state["texts"] = {
        "kant_sample": (
            "The categorical imperative is the central philosophical concept in "
            "Kant's deontological moral philosophy. It is a way of evaluating "
            "motivations for action. The categorical imperative is Kant's "
            "formulation of the moral law that he believes is binding on all "
            "rational beings regardless of empirical considerations."
        ),
        "nietzsche_sample": (
            "God is dead. God remains dead. And we have killed him. How shall we "
            "comfort ourselves, the murderers of all murderers? What was holiest "
            "and mightiest of all that the world has yet owned has bled to death "
            "under our knives: who will wipe this blood off us?"
        ),
        "hume_sample": (
            "Reason is, and ought only to be the slave of the passions, and can "
            "never pretend to any other office than to serve and obey them. The "
            "essence of belief is some sentiment or feeling that does not depend "
            "on the will, and which accompanies the idea whenever it is present."
        ),
    }
    return {"texts": {k: v for k, v in _state["texts"].items()}}

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@router.post("/analyze")
async def run_analysis(params: AnalysisParams):
    if not _state["texts"]:
        raise HTTPException(400, "No texts loaded. Upload or paste texts first.")

    # Check if inputs are unchanged since last analysis (cache hit)
    params_dict = {"lsa_components": params.lsa_components, "coherence_window": params.coherence_window}
    current_hash = _compute_analysis_hash(_state["texts"], params_dict)

    if current_hash == _state.get("analysis_hash") and _state["results"] is not None:
        logger.info("Analysis cache hit — returning previous results")
        results = _state["results"]
        results_dict = results.to_dict(orient="records") if hasattr(results, "to_dict") else results
        return {
            "results": results_dict,
            "viz_data": _serialize_viz(_state["viz_data"]),
            "cached": True,
        }

    try:
        analyzer = IntegratedPhilosophicalAnalyzer(
            lsa_components=params.lsa_components,
            coherence_window=params.coherence_window,
        )
        analyzer.fit(_state["texts"])
        results = analyzer.analyze_multiple_texts(_state["texts"])

        _state["results"] = results

        viz_gen = VisualizationGenerator(output_dir="reports/visualizations")
        viz_data = viz_gen.generate_all_visualizations(
            analysis_results=results,
            texts=_state["texts"],
        )
        _state["viz_data"] = viz_data
        _state["analysis_hash"] = current_hash

        # Convert DataFrame to JSON-safe dict
        results_dict = results.to_dict(orient="records") if hasattr(results, "to_dict") else results

        return {
            "results": results_dict,
            "viz_data": _serialize_viz(viz_data),
            "cached": False,
        }
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(500, f"Analysis failed: {e}")

# ---------------------------------------------------------------------------
# Visualization data
# ---------------------------------------------------------------------------

@router.get("/results")
async def get_results():
    if _state["results"] is None:
        raise HTTPException(404, "No results available")
    results = _state["results"]
    return results.to_dict(orient="records") if hasattr(results, "to_dict") else results


@router.get("/viz/dashboard")
async def get_dashboard():
    vd = _state.get("viz_data")
    if not vd or "dashboard" not in vd:
        raise HTTPException(404, "Dashboard data not available")
    return vd["dashboard"]


@router.get("/viz/temporal")
async def get_temporal():
    vd = _state.get("viz_data")
    if not vd or "temporal" not in vd:
        raise HTTPException(404, "Temporal data not available")
    return vd["temporal"]


@router.get("/viz/network")
async def get_network():
    vd = _state.get("viz_data")
    if not vd or "network" not in vd:
        raise HTTPException(404, "Network data not available")
    return vd["network"]


@router.get("/viz/all")
async def get_all_viz():
    vd = _state.get("viz_data")
    if not vd:
        raise HTTPException(404, "No visualization data available")
    return _serialize_viz(vd)


def _serialize_viz(viz_data: dict) -> dict:
    """Ensure viz data is JSON-serializable."""
    import numpy as np

    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    return _convert(viz_data)
