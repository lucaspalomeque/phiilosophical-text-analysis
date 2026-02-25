# Test Coverage

## Summary: 115 tests

| Directory | Files | Tests | Description |
|-----------|-------|-------|-------------|
| `tests/core/` | 4 | ~20 | Coherence analyzer, POS analyzer, convex hull, integrated analyzer |
| `tests/visualization/` | 1 | ~8 | Dashboard data generation, temporal data, JSON output |
| `tests/web/test_api.py` | 1 | ~33 | All FastAPI endpoints: health, text CRUD, upload, samples, analysis, viz data, SPA routing |
| `tests/web/test_viz_contract.py` | 1 | ~17 | Frontend/backend data contract: dashboard metrics, temporal arrays, network nodes/links, results format |
| `tests/test_math_validation.py` | 1 | ~30 | Cosine similarity axioms, coherence ordering, second-order formula, temporal coherence, determiner counting, Cohen's d, t-stat vs scipy, coherence decay, syntactic complexity, LSA sanity, end-to-end consistency |
| `tests/test_full_pipeline.py` | 1 | ~7 | End-to-end pipeline with edge cases (short texts, repetitive content, number-heavy) |
| **Total** | **9** | **~115** | |

## Running tests

```bash
# All tests
PYTHONPATH=src python3 -m pytest tests/ -v

# Fast unit tests only (no NLTK downloads, no full pipeline)
PYTHONPATH=src python3 -m pytest tests/ -v -m "not integration"

# Integration tests only (full pipeline including NLTK)
PYTHONPATH=src python3 -m pytest tests/ -v -m "integration"

# Web API tests only
PYTHONPATH=src python3 -m pytest tests/web/ -v

# Core analysis tests only
PYTHONPATH=src python3 -m pytest tests/core/ -v

# Math validation tests only
PYTHONPATH=src python3 -m pytest tests/test_math_validation.py -v

# With coverage report
PYTHONPATH=src python3 -m pytest --cov=src/philosophical_analysis --cov-report=html tests/
```

## Test dependencies

- pytest >= 6.2.0
- pytest-cov >= 3.0.0
- httpx >= 0.24.0 (required for FastAPI TestClient)

## Key fixtures (conftest.py)

- `temp_data_dir` — Session-scoped temporary directory for test data
- `sample_philosophical_texts` — Kant, Hume, Nietzsche, and incoherent style texts
- `mock_nltk_data` — Session-scoped fixture that downloads NLTK data to a temp directory; sets `NLTK_DATA` env var and prepends to `nltk.data.path`

## Test gaps

1. No performance benchmarks (execution time, memory on large corpora)
2. No fuzz testing on text input (unicode edge cases, very long texts)
3. No concurrent request tests for the web API
4. Semantic network generation has basic coverage only
