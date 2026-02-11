"""
Frontend Server — Serves static files and proxies API requests to the backend.
Only this container is exposed to the outside (port 3000).
"""

import httpx
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL = "http://backend:8000"


# ---------------------------------------------------------------------------
# Middleware to disable caching for static files (dev mode)
# ---------------------------------------------------------------------------
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        if request.url.path.endswith((".js", ".css", ".html")) or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            # Remove headers that allow 304 responses
            if "etag" in response.headers:
                del response.headers["etag"]
            if "last-modified" in response.headers:
                del response.headers["last-modified"]
        return response

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Casting Classifier — Frontend")
app.add_middleware(NoCacheMiddleware)


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


@app.post("/api/similar")
async def proxy_similar(file: UploadFile = File(...)):
    """Proxy similarity search request to the backend."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            file_content = await file.read()
            response = await client.post(
                f"{BACKEND_URL}/api/similar",
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


@app.get("/api/images/{path:path}")
async def proxy_image(path: str):
    """Proxy image serving from backend casting_data directory."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{BACKEND_URL}/api/images/{path}")
        if response.status_code == 200:
            return Response(
                content=response.content,
                media_type=response.headers.get("content-type", "image/jpeg"),
                status_code=200,
            )
        return JSONResponse(
            content={"detail": "Image not found"},
            status_code=response.status_code,
        )
    except Exception as e:
        return JSONResponse(
            content={"detail": f"Proxy error: {str(e)}"},
            status_code=502,
        )


# ---------------------------------------------------------------------------
# Static files — served AFTER api routes so /api/* takes priority
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
