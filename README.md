# Philosophical Text Analysis

Automated analysis of philosophical texts using psycholinguistic techniques from [Bedi et al. (2015)](https://www.nature.com/articles/npjschz201530). Applies LSA-based coherence analysis, POS-based syntactic metrics, and convex hull classification to philosophical works.

## Quick start

```bash
# Web app (recommended)
pip install -e ".[web]"
PYTHONPATH=src uvicorn philosophical_analysis.web.app:app --reload
# Open http://localhost:8000

# CLI
PYTHONPATH=src python -m philosophical_analysis.cli analyze --text your_text.txt

# Docker
docker build -t philo-analysis .
docker run -p 8000:8000 philo-analysis

# Tests (115 total)
PYTHONPATH=src python -m pytest tests/ -v
```

## Architecture

```
src/philosophical_analysis/
  core/
    enhanced_coherence.py   # LSA coherence: 1st/2nd order, temporal, decay
    pos_analyzer.py         # POS tagging, target determiners (Bedi spec)
    integrated_analyzer.py  # Orchestrator: POS + Coherence + Classifier
    convex_hull.py          # Convex hull classification
    analyzer.py             # Basic LSA + TF-IDF coherence
  visualization/
    generator.py            # JSON data generation for frontend
    semantic_network.py     # Concept extraction + co-occurrence graph
    config/default.py       # VIZ_CONFIG (currently unused, see Known Issues)
  web/
    app.py                  # FastAPI backend with security headers middleware
    routes/api.py           # REST endpoints + analysis caching
    static/css/             # Apple-inspired glassmorphism design system
    static/js/app.js        # SPA router + Plotly/D3 chart renderers
    static/js/api-client.js # Fetch wrapper for all API calls
    templates/index.html    # Single-page entry point
  cli.py                    # CLI interface (analyze, batch)
  cli_extensions.py         # CLI visualization commands
```

## Analysis pipeline

```
Raw text
  -> Preprocessing (NLTK tokenize, lemmatize, stopwords)
  -> TF-IDF vectorization (max_features adaptive, ngram 1-2)
  -> LSA via TruncatedSVD (default 10 components)
  -> Sentence vectors -> Cosine similarity
  -> Coherence metrics (1st order, 2nd order, temporal, decay)
  -> POS features (target determiners, phrase lengths, syntactic complexity)
  -> Convex hull classification
  -> DataFrame of results -> VisualizationGenerator -> JSON
  -> Frontend renders via Plotly.js / D3.js
```

### Key metrics

| Metric | Description | Range |
|--------|-------------|-------|
| First-order coherence | Mean cosine similarity between consecutive LSA sentence vectors | [0, 1] |
| Second-order coherence | Stability of first-order scores: `1 - mean(\|delta\|)` | [0, 1] |
| Temporal coherence | Sliding-window first-order coherence (window=5) | [0, 1] |
| Coherence decay rate | Exponential decay of coherence with sentence distance | >= 0 |
| Target determiners freq | Frequency of {that, what, whatever, which, whichever} per Bedi et al. | [0, 1] |
| Syntactic complexity | Dict with avg_sentence_length, clauses_per_sentence, POS diversity | dict |
| Statistical test | One-sample t-test vs baseline 0.3; Cohen's d effect size | p-value, d |

### Minimum data requirements

- `EnhancedCoherenceAnalyzer.fit()` requires **>= 10 sentences** across all texts combined
- Individual text analysis requires **>= 2 sentences**
- LSA quality degrades severely with < 100 words per text
- POS analyzer has NLTK fallbacks (regex tokenizer + heuristic tagger) if NLTK data is unavailable

## Web application

**Stack**: FastAPI + vanilla HTML/CSS/JS (no build tools)

**Design**: Apple-inspired glassmorphism — black background, brushed gold (#C9A96E), frosted glass cards

**Charts**: Plotly.js 2.27.0 for statistical charts, D3.js 7.8.5 for force-directed network

**Routing**: Hash-based SPA with 5 views:
- `#/upload` — Upload files, paste text, or load samples; configure LSA params; run analysis
- `#/dashboard` — Metric cards, radar/bar/scatter/heatmap charts, detailed table
- `#/temporal` — Coherence timeline per philosopher, trend analysis, decay visualization
- `#/network` — D3 force-directed semantic network with filters (philosopher, category, strength)
- `#/compare` — Parallel coordinates, small multiples (radar), ranked dot plot, ridgeline

**API endpoints**:
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/texts` | List loaded texts |
| POST | `/api/texts` | Add text (name + content) |
| DELETE | `/api/texts/{name}` | Remove text |
| POST | `/api/upload` | Upload .txt files |
| POST | `/api/samples` | Load sample texts |
| POST | `/api/analyze` | Run analysis (cached if inputs unchanged) |
| GET | `/api/results` | Raw analysis results |
| GET | `/api/viz/dashboard` | Dashboard viz data |
| GET | `/api/viz/temporal` | Temporal viz data |
| GET | `/api/viz/network` | Network viz data |
| GET | `/api/viz/all` | All viz data combined |

**Caching**: Server-side SHA-256 hash of texts + params. Re-analysis is skipped when inputs are unchanged. Cache is invalidated on any text add/remove/upload.

**Security headers**: X-Content-Type-Options, X-Frame-Options, Referrer-Policy, Permissions-Policy.

## Testing

```bash
# All tests
PYTHONPATH=src python -m pytest tests/ -v

# Fast unit tests only
PYTHONPATH=src python -m pytest tests/ -v -m "not integration"

# Integration tests only
PYTHONPATH=src python -m pytest tests/ -v -m "integration"

# Web API tests only
PYTHONPATH=src python -m pytest tests/web/ -v

# Math validation tests only
PYTHONPATH=src python -m pytest tests/test_math_validation.py -v
```

### Test suite (115 tests)

| Directory | Tests | What it covers |
|-----------|-------|----------------|
| `tests/core/` | ~20 | Coherence analyzer, POS analyzer, convex hull, integrated analyzer |
| `tests/visualization/` | ~8 | Dashboard data generation, temporal data, JSON output |
| `tests/web/test_api.py` | ~33 | All FastAPI endpoints: CRUD, upload, analysis, viz data, SPA routing |
| `tests/web/test_viz_contract.py` | ~17 | Frontend/backend data contract validation |
| `tests/test_math_validation.py` | ~30 | Cosine similarity axioms, coherence ordering, second-order formula, temporal coherence, determiner counting, Cohen's d, t-stat vs scipy, coherence decay, syntactic complexity, LSA sanity, end-to-end consistency |
| `tests/test_full_pipeline.py` | ~7 | End-to-end pipeline with edge cases |

### CI

GitHub Actions runs on push/PR to main: Python 3.9 + 3.11 matrix, flake8 linting, unit tests, integration tests.

## Dependencies

- **Core**: numpy, pandas, scipy, scikit-learn, nltk, networkx, matplotlib, seaborn
- **Web**: fastapi, uvicorn, python-multipart, jinja2, aiofiles
- **Dev**: pytest, pytest-cov, httpx, flake8, black, isort, mypy
- **Frontend (CDN)**: Plotly.js 2.27.0, D3.js 7.8.5

## Known issues and technical debt

### Critical (fix before multi-user deployment)

1. **Global shared state** — `_state` in `api.py` is a module-level dict shared across all requests. All users see the same texts and results. No per-user sessions. Safe for single-user local use; must be replaced with session-based state (e.g., UUID tokens or server-side sessions) before deploying publicly.

### Medium priority

2. **Sample texts too short** — The built-in samples (`/api/samples`) have only 2-4 sentences each (45-51 words). LSA needs >= 10 sentences combined for `fit()`. Demo results are statistically unreliable. Should be expanded to 250+ words per sample.

3. **No input validation on text length** — Users can submit 1-word texts. The analyzer returns a 500 error or zeroed metrics silently. Should validate minimum 3 sentences (~50 words) before accepting, and enforce a maximum length to prevent memory issues.

4. **No export/download** — Web UI has no way to download results as CSV/JSON. The `/api/results` endpoint returns JSON but there's no download button in the UI.

5. **VIZ_CONFIG unused** — `visualization/config/default.py` defines a `VIZ_CONFIG` dict (theme colors, network params, temporal settings) that is imported but never actually referenced in `generator.py`. Colors are hardcoded instead.

### Low priority

6. **Dead imports in generator.py** — `Counter`, `networkx as nx`, `cosine_similarity`, `IntegratedPhilosophicalAnalyzer` are imported but never used.

7. **No raw data view** — The web UI shows aggregated charts and tables but no way to see per-sentence coherence scores, feature vectors, or raw JSON.

8. **Coherence thresholds not empirically validated** — The thresholds in `integrated_analyzer.py` (0.6 = "highly coherent", 0.4 = "moderately coherent") are educated guesses, not validated against a ground truth dataset. See CLAUDE.md for the full threshold table.

9. **LSA random state not set** — `TruncatedSVD` in `enhanced_coherence.py` doesn't set `random_state`, making results non-reproducible across runs.

## Project files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Developer guide for Claude Code: coding conventions, data science standards, detailed metric definitions, statistical rigor checklist |
| `ARCHITECTURE.md` | High-level architecture description (in Spanish) |
| `tests/TEST_COVERAGE.md` | Test coverage details per component |
| `Dockerfile` | Production container with NLTK data pre-downloaded |
| `.github/workflows/ci.yml` | CI pipeline: lint + unit tests + integration tests |
| `setup.py` | Package config with extras: `[dev]`, `[web]`, `[viz]`, `[notebook]`, `[all]` |

## Scientific basis

Based on: [Automated analysis of free speech predicts psychosis onset in high-risk youths](https://www.nature.com/articles/npjschz201530) (Bedi et al., 2015, *npj Schizophrenia*)

The paper's methodology — LSA coherence, target determiners, convex hull classification — is faithfully implemented but applied to philosophical texts (not clinical data), so clinical thresholds may not transfer directly.

## License

MIT
