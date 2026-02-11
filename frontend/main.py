"""
Frontend Server — Serves static files and proxies API requests to the backend.
Only this container is exposed to the outside (port 3000).
"""

import httpx
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL = "http://backend:8000"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Casting Classifier — Frontend")


# ---------------------------------------------------------------------------
# Proxy endpoints — forward to backend (not exposed externally)
# ---------------------------------------------------------------------------
@app.post("/api/classify")
async def proxy_classify(file: UploadFile = File(...)):
    """Proxy the classify request to the backend."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            file_content = await file.read()
            response = await client.post(
                f"{BACKEND_URL}/api/classify",
                files={"file": (file.filename, file_content, file.content_type)},
            )
        return JSONResponse(
            content=response.json(),
            status_code=response.status_code,
        )
    except httpx.ConnectError:
        return JSONResponse(
            content={"detail": "Backend service unavailable"},
            status_code=503,
        )
    except Exception as e:
        return JSONResponse(
            content={"detail": f"Proxy error: {str(e)}"},
            status_code=502,
        )


@app.get("/api/health")
async def proxy_health():
    """Proxy health check to backend."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{BACKEND_URL}/api/health")
        return JSONResponse(
            content=response.json(),
            status_code=response.status_code,
        )
    except Exception:
        return JSONResponse(
            content={"status": "backend_unavailable"},
            status_code=503,
        )


# ---------------------------------------------------------------------------
# Static files — served AFTER api routes so /api/* takes priority
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
