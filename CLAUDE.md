# Philosophical Text Analysis — Project Guide

## Overview

Automated analysis of philosophical texts using psycholinguistic techniques from Bedi et al. (2015) — "Automated analysis of free speech predicts psychosis onset." Applies LSA-based coherence analysis, POS-based syntactic metrics, and convex hull classification to philosophical works (Kant, Nietzsche, Hume, and others).

## Architecture

```
src/philosophical_analysis/
├── core/                         # Analysis engine
│   ├── analyzer.py               # Basic LSA + TF-IDF coherence
│   ├── enhanced_coherence.py     # 1st/2nd order, temporal, decay metrics
│   ├── integrated_analyzer.py    # Orchestrator: POS + Coherence + Classifier
│   ├── pos_analyzer.py           # Target determiners, phrase metrics (Bedi spec)
│   └── convex_hull.py            # Convex hull classification
├── visualization/
│   ├── generator.py              # JSON data generation for frontend
│   ├── semantic_network.py       # Concept extraction + co-occurrence graph
│   ├── config/default.py         # Theme colors (gold palette)
│   └── templates/                # Legacy HTML visualizations (deprecated)
├── web/
│   ├── app.py                    # FastAPI backend (SPA)
│   ├── routes/api.py             # REST endpoints
│   ├── static/css/               # Apple-inspired glassmorphism design system
│   ├── static/js/app.js          # SPA router + Plotly/D3 chart renderers
│   └── templates/index.html      # Single-page entry point
└── cli.py, cli_extensions.py     # CLI interface
```

## Running the project

```bash
# Web app
PYTHONPATH=src python3 -m uvicorn philosophical_analysis.web.app:app --reload

# CLI
PYTHONPATH=src python3 -m philosophical_analysis.cli analyze --help
```

## Analysis pipeline

```
Raw text → Preprocessing (NLTK tokenize, lemmatize, stopwords)
         → TF-IDF vectorization (max_features adaptive, ngram 1-2)
         → LSA via TruncatedSVD (default 10 components)
         → Sentence vectors → Cosine similarity
         → Coherence metrics (1st order, 2nd order, temporal, decay)
         → POS features (determiners, phrase lengths)
         → Convex hull classification
         → DataFrame of results → VisualizationGenerator → JSON
```

## Data science standards

### Coherence metrics

- **First-order coherence**: mean cosine similarity between consecutive LSA sentence vectors. Valid range: [0, 1]. Values near 0 indicate semantic disconnection; above 0.6 is "highly coherent."
- **Second-order coherence**: stability of first-order scores. Computed as `1 - mean(|Δcoherence|)`. Measures consistency, not level.
- **Temporal coherence**: sliding-window first-order coherence (default window=5). Trend computed via Pearson correlation of window position vs. coherence.
- **Coherence decay rate**: how coherence drops with sentence distance. Fit as exponential decay `a * exp(-b * d)`.
- **Statistical test**: one-sample t-test against baseline mean 0.3. Report p-value and Cohen's d effect size. No multiple comparison correction currently applied.

### Known thresholds (hardcoded, need empirical validation)

| Metric | Threshold | Label | Source |
|--------|-----------|-------|--------|
| Coherence | > 0.6 | highly coherent | integrated_analyzer.py |
| Coherence | > 0.4 | moderately coherent | integrated_analyzer.py |
| Coherence | > 0.3 | coherent (basic) | analyzer.py |
| 2nd order | > 0.7 | highly stable | integrated_analyzer.py |
| Determiners freq | > 0.015 | high usage | integrated_analyzer.py |
| Determiners freq | > 0.008 | moderate usage | integrated_analyzer.py |
| Complexity score | > 1.5 | high complexity | integrated_analyzer.py |
| t-test baseline | 0.3 | null hypothesis mean | enhanced_coherence.py |

These thresholds are **not empirically validated** against a ground truth dataset. They should be treated as exploratory, not confirmatory.

### When modifying analysis code

- Always preserve the `window_coherences` list in temporal analysis — the frontend reads it directly for timeline charts
- The `generate_all_visualizations()` output format is the API contract with the frontend; changes require updating both `routes/api.py` and `app.js`
- Sentence preprocessing must keep minimum 3 words per sentence (enhanced_coherence) or 2 words (basic analyzer) — these differ and should be unified
- TF-IDF uses `min_df=1` which is very permissive; consider raising for larger corpora
- LSA components default to 10, adaptive to corpus size via `min(n, features-1, docs-1)`

### Statistical rigor checklist

When adding or modifying statistical analyses:

1. **Report effect sizes**, not just p-values. Cohen's d is already computed for coherence.
2. **Document assumptions**: normality, independence, equal variance. Current t-tests assume normality without testing it.
3. **Confidence intervals** are not yet computed for any metric — they should be.
4. **Multiple comparisons**: when comparing N philosophers, apply Bonferroni or FDR correction. Currently absent.
5. **Sample size**: the convex hull classifier needs `dimensions + 1` points minimum. Short texts may fail silently.
6. **Reproducibility**: LSA with TruncatedSVD uses random state — set `random_state=42` for reproducible results. Currently not set in enhanced_coherence.py.

### Semantic network

- Concept extraction uses a **curated list of ~90 philosophical terms** across 7 categories (epistemological, metaphysical, ethical, aesthetic, political, religious, logical)
- Co-occurrence uses sentence-level + 5-word proximity window with log-probability scaling
- Network density, degree distribution, and clustering coefficient should be reported
- No word sense disambiguation — "object" (philosophy) vs "object" (programming) are conflated

## Frontend

- **Stack**: FastAPI + vanilla HTML/CSS/JS (no build tools)
- **Design**: Apple-inspired glassmorphism — black bg, brushed gold (#C9A96E) titles, light grey (#E5E5E7) text
- **Charts**: Plotly.js for statistical charts, D3.js for force-directed network
- **Routing**: Hash-based SPA (`#/upload`, `#/dashboard`, `#/temporal`, `#/network`, `#/compare`)
- **Data flow**: Frontend fetches JSON from `/api/viz/*` endpoints, renders client-side

### When modifying the frontend

- All CSS uses custom properties defined in `design-system.css` — never hardcode colors
- Plotly charts use `plotlyDefaults()` helper for consistent theming
- The `CHART_COLORS` array maps to philosophers by index order — the dashboard assigns colors to philosophers in the order they appear in the JSON
- D3 network uses category-based coloring: metaphysics=gold, ethics=cyan, epistemology=grey, logic=purple

## Testing

```bash
# All tests (115 total)
PYTHONPATH=src python3 -m pytest tests/ -v

# Fast unit tests only
PYTHONPATH=src python3 -m pytest tests/ -v -m "not integration"

# Integration tests only (runs full analysis pipeline)
PYTHONPATH=src python3 -m pytest tests/ -v -m "integration"

# Web API tests only
PYTHONPATH=src python3 -m pytest tests/web/ -v
```

- `tests/core/` — unit tests for each analyzer component
- `tests/visualization/` — visualization data generation
- `tests/web/test_api.py` — FastAPI endpoint tests (text CRUD, upload, analysis, viz data, SPA routing)
- `tests/web/test_viz_contract.py` — validates viz data structures match what frontend JS expects
- `tests/test_full_pipeline.py` — integration: texts → analysis → viz JSON
- Test fixtures use synthetic philosophical texts (Kant/Nietzsche/Hume style)
- Web tests use FastAPI TestClient (requires `httpx`)

## Dependencies

- Core: numpy, pandas, scipy, scikit-learn, nltk, networkx
- Web: fastapi, uvicorn, python-multipart, jinja2, aiofiles
- Dev: pytest, pytest-cov, httpx, flake8, black, isort, mypy
- Viz: plotly, matplotlib, seaborn
- Frontend: Plotly.js 2.27.0, D3.js 7.8.5 (CDN)

## Key decisions

- **FastAPI over Streamlit**: full CSS control for the Apple-inspired design
- **Vanilla JS over React**: 5 views, no build tools, Python-friendly team
- **JSON interchange**: Python generates JSON, JS renders charts — clean separation
- **Bedi et al. methodology**: target determiners, LSA coherence, convex hull — faithfully implemented but applied to philosophical (not clinical) texts, so thresholds may not transfer

## Project language

- Code: English
- Documentation (ARCHITECTURE.md): Spanish
- Commit messages: English
