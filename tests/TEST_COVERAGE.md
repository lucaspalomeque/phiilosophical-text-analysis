# Test Coverage Documentation

## Overview

This document provides an overview of the test coverage for the Philosophical Text Analysis project, highlighting what's tested, what's not, and recommendations for further testing.

## Current Test Coverage

### Core Components

| Component | Coverage | Description |
|-----------|----------|-------------|
| `enhanced_coherence.py` | Good | Unit tests verify initialization, fitting, analysis pipeline, and error handling |
| `pos_analyzer.py` | Good | Unit tests cover POS tagging, determiner analysis, and phrase extraction |
| `convex_hull.py` | Good | Tests for classification functionality and edge cases |
| `integrated_analyzer.py` | Good | Tests verify component integration, end-to-end analysis, and cross-validation |

### Visualization Components

| Component | Coverage | Description |
|-----------|----------|-------------|
| `generator.py` | Good | Tests for dashboard data generation, temporal data creation, JSON output, and error handling |
| `semantic_network.py` | Basic | Basic test for network generation, could use more comprehensive tests |

### Web API (FastAPI)

| Component | Coverage | Description |
|-----------|----------|-------------|
| Health endpoint | Good | Verifies `/api/health` returns 200 |
| Text management | Good | CRUD for texts: add, get, remove, upload files, load samples |
| File upload | Good | Single/multiple file upload, extension stripping, UTF-8 validation |
| Analysis endpoint | Good | Tests empty state error, full pipeline with samples, state persistence |
| Viz data endpoints | Good | 404 before analysis, correct data after analysis for dashboard/temporal/network/all |
| SPA routing | Good | Index, catch-all, static/api paths not caught |

### Viz Data Contract Tests

| Contract | Coverage | Description |
|-----------|----------|-------------|
| Dashboard | Good | Validates philosopher metrics dict, stats fields, JSON serializability |
| Temporal | Good | Validates timeline arrays are numeric lists, avg_coherence present |
| Network | Good | Validates nodes/links structure, source/target on links, metadata |
| Results | Good | Validates list-of-dicts format, text_id and coherence fields |

### Integration Tests

| Test Type | Coverage | Description |
|-----------|----------|-------------|
| Full Pipeline | Good | End-to-end test from text input through analysis to visualization output |
| Edge Cases | Good | Tests with short texts, repetitive content, and number-heavy content |
| Web API Pipeline | Good | Samples -> analyze -> viz endpoints, verified via contract tests |

## Test Summary

| Directory | Files | Tests |
|-----------|-------|-------|
| `tests/core/` | 4 | ~20 |
| `tests/visualization/` | 1 | ~8 |
| `tests/web/` | 2 | ~50 |
| `tests/` (root) | 2 | ~9 |
| **Total** | **9** | **~85** |

## Test Gaps and Recommendations

1. **Semantic Network Analysis**: Limited testing of concept extraction and relationship building
2. **Performance Tests**: No benchmarks for execution time or memory on large text collections
3. **Configuration Handling**: Limited testing of config file loading and parameter overriding

## Running Tests

```bash
# Run all tests
PYTHONPATH=src python3 -m pytest tests/ -v

# Run only unit tests (fast)
PYTHONPATH=src python3 -m pytest tests/ -v -m "not integration"

# Run only integration tests
PYTHONPATH=src python3 -m pytest tests/ -v -m "integration"

# Run only web API tests
PYTHONPATH=src python3 -m pytest tests/web/ -v

# Run only core tests
PYTHONPATH=src python3 -m pytest tests/core/ -v

# Generate HTML coverage report
PYTHONPATH=src python3 -m pytest --cov=philosophical_analysis --cov-report=html tests/
```

## Test Dependencies

- pytest>=6.2.0
- pytest-cov>=3.0.0
- httpx>=0.24.0 (required for FastAPI TestClient)

## Continuous Integration

GitHub Actions CI runs on every push/PR to main:
- Python 3.9 and 3.11 matrix
- Linting with flake8
- Unit tests (fast, no external dependencies)
- Integration tests (full pipeline including NLTK)
- Web dependencies installed via `pip install -e ".[dev,web]"`
