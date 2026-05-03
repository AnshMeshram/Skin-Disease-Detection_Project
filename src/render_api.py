from __future__ import annotations

import asyncio
import importlib
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Skin Disease Detection API", version="1.0-render")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_api_module: Any | None = None
_api_app: Any | None = None
_loading = False
_error: str | None = None


def _load_backend_sync() -> None:
    global _api_module, _api_app, _loading, _error
    _loading = True
    _error = None
    try:
        module = importlib.import_module("src.api")
        module._load_model_sync()
        _api_module = module
        _api_app = module.app
    except Exception as exc:
        _api_module = None
        _api_app = None
        _error = str(exc)
        print(f"Render API backend load failed: {exc}")
    finally:
        _loading = False


@app.on_event("startup")
async def start_backend_loader() -> None:
    asyncio.create_task(asyncio.to_thread(_load_backend_sync))


@app.get("/health")
async def health() -> dict[str, Any]:
    model_loaded = False
    model_loading = _loading
    model_error = _error

    if _api_module is not None:
        model_loaded = getattr(_api_module, "_model", None) is not None
        model_loading = bool(getattr(_api_module, "_model_loading", False))
        model_error = getattr(_api_module, "_model_error", None)

    return {
        "status": "ok",
        "backend_ready": _api_app is not None,
        "model_loaded": model_loaded,
        "model_loading": model_loading,
        "model_error": model_error,
        "classes": 9,
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def forward_to_backend(path: str, request: Request) -> Response:
    if _api_app is None:
        raise HTTPException(
            status_code=503,
            detail="Backend is still loading. Refresh /health until backend_ready is true.",
        )

    body = await request.body()
    response_start: dict[str, Any] | None = None
    response_body = bytearray()
    received = False

    scope = dict(request.scope)
    scope["path"] = f"/{path}"
    scope["root_path"] = ""
    scope["app"] = _api_app

    async def receive() -> dict[str, Any]:
        nonlocal received
        if received:
            return {"type": "http.request", "body": b"", "more_body": False}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        nonlocal response_start
        if message["type"] == "http.response.start":
            response_start = message
        elif message["type"] == "http.response.body":
            response_body.extend(message.get("body", b""))

    await _api_app(scope, receive, send)

    if response_start is None:
        raise HTTPException(status_code=500, detail="Backend response did not start")

    headers = {
        key.decode("latin-1"): value.decode("latin-1")
        for key, value in response_start.get("headers", [])
        if key.lower() not in {b"content-length", b"transfer-encoding"}
    }
    return Response(
        content=bytes(response_body),
        status_code=int(response_start["status"]),
        headers=headers,
    )
