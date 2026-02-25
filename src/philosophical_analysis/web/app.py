"""
Philosophical Text Analysis — Web Application

FastAPI-based web interface serving a unified SPA with Apple-inspired
glassmorphism design. Replaces the former Streamlit interface.

Run with:
    uvicorn philosophical_analysis.web.app:app --reload
    # or
    python -m philosophical_analysis.web.app
"""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from .routes.api import router as api_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Philosophical Text Analysis",
    description="Analyze philosophical texts using psycholinguistic techniques",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response

# Mount static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Register API routes
app.include_router(api_router)

# ---------------------------------------------------------------------------
# SPA catch-all: serve index.html for all non-API, non-static routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the SPA entry point."""
    index_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/{path:path}", response_class=HTMLResponse)
async def spa_catchall(path: str):
    """Catch-all route to support client-side routing."""
    # If the path looks like a static file, let it 404 naturally
    if path.startswith("static/") or path.startswith("api/"):
        return HTMLResponse(status_code=404, content="Not found")
    index_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run the web app via uvicorn."""
    import uvicorn
    uvicorn.run(
        "philosophical_analysis.web.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
