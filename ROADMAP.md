# Roadmap

## Completed

### Core analysis pipeline
- LSA-based semantic coherence (1st order, 2nd order, temporal, decay)
- POS analysis with target determiners per Bedi et al. (2015)
- Convex hull classification with leave-one-out cross-validation
- Integrated analyzer orchestrating all components
- NLTK fallbacks (regex tokenizer + heuristic POS tagger) when NLTK data unavailable

### Web application
- FastAPI SPA with Apple-inspired glassmorphism design (black + gold)
- 5 views: Upload, Dashboard, Temporal, Network, Compare
- Plotly.js charts (radar, bar, scatter, heatmap, parallel coordinates, violin)
- D3.js force-directed semantic network with filtering
- Server-side analysis caching (SHA-256 hash)
- Security headers middleware
- Skeleton loading states, toast notifications, keyboard navigation

### Testing & CI
- 115 tests across core, visualization, web API, math validation, and integration
- GitHub Actions CI: Python 3.9 + 3.11, flake8, unit + integration tests
- Math validation suite verifying all formulas against hand computations and scipy

### Infrastructure
- Dockerfile with NLTK data pre-downloaded
- CLI with analyze and batch commands
- Package installable via `pip install -e ".[web]"`

## Open issues (prioritized)

See README.md "Known issues and technical debt" for the full list. Top 3:

1. **Global shared state** — Must add per-user sessions before multi-user deployment
2. **Sample texts too short** — Expand to 250+ words for reliable LSA demos
3. **No input validation on text length** — Add min/max checks with clear error messages

## Future ideas (not started)

- Export/download results as CSV/JSON from web UI
- Per-sentence coherence drill-down view
- Wire `VIZ_CONFIG` into visualization generator (replace hardcoded colors)
- Set `random_state=42` on TruncatedSVD for reproducible LSA
- Multiple comparison correction (Bonferroni/FDR) when comparing N philosophers
- Cross-language coherence analysis
- Philosophical theme auto-classification (epistemology/ethics/metaphysics)
- Argumentative structure detection (premise-conclusion)
