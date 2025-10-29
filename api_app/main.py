import os
import io
import shutil
import uuid
import time
import hashlib
import logging
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRouter
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse

from .pipeline import process_image_request
from .config import settings
from .security import verify_jwt_token, create_jwt
from .limiter import rate_limit
from .cache import get as cache_get, set as cache_set
from .logging_setup import configure_logging
from .schemas import ProcessQuery, ProcessResponse


configure_logging()
logger = logging.getLogger("api")

app = FastAPI(title="Garment Measurement API", version="1.0.0")

# CORS (optional local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix=settings.API_PREFIX, tags=["v1"])


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    resp = None
    try:
        resp = await call_next(request)
        return resp
    finally:
        duration_ms = int((time.time() - start) * 1000)
        logger.info(
            "request",
            extra={
                "path": str(request.url.path),
                "method": request.method,
                "status": getattr(resp, "status_code", 0),
                "duration_ms": duration_ms,
            },
        )


@router.get("/health", summary="Liveness/Readiness probe")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.post("/auth/token")
def issue_token() -> Dict[str, str]:
    token = create_jwt("local-user", ttl_seconds=3600)
    return {"token": token}


def _hash_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _cache_payload_key(img_hash: str, q: ProcessQuery) -> Dict[str, Any]:
    return {
        "img": img_hash,
        "category_id": int(q.category_id),
        "true_size": q.true_size,
        "true_waist": float(q.true_waist) if q.true_waist is not None else None,
        "unit": q.unit.lower(),
    }


@router.post("/process", response_model=ProcessResponse, dependencies=[Depends(verify_jwt_token)])
async def process(
    request: Request,
    background: BackgroundTasks,
    image: UploadFile = File(...),
    category_id: int = Form(...),
    true_size: str = Form(...),
    true_waist: Optional[float] = Form(None),
    unit: str = Form("cm"),
    webhook_url: Optional[str] = Form(None),
):
    rate_limit(request)
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    # Validate fields via schema
    q = ProcessQuery(category_id=category_id, true_size=true_size, true_waist=true_waist, unit=unit)

    # Caching
    content = await image.read()
    img_hash = _hash_bytes(content)
    cache_key = _cache_payload_key(img_hash, q)
    cached = cache_get(cache_key)
    if cached:
        if webhook_url:
            background.add_task(lambda: requests.post(webhook_url, json=cached, timeout=10))
        return cached

    # Workspace
    request_id = str(uuid.uuid4())
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    work_dir = os.path.join(base_dir, "api_runs", request_id)
    os.makedirs(work_dir, exist_ok=True)

    input_image_path = os.path.join(work_dir, "image.jpg")
    with open(input_image_path, "wb") as f:
        f.write(content)

    # Run pipeline
    try:
        result = process_image_request(
            input_image_path=input_image_path,
            work_dir=work_dir,
            category_id=q.category_id,
            true_size=q.true_size,
            true_waist=q.true_waist,
            unit=q.unit,
        )
    except HTTPException:
        raise
    except Exception as e:
        # Log full traceback for diagnosis
        logger.exception("pipeline_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Processing failed")

    # Cache and optional webhook
    cache_set(cache_key, result)
    if webhook_url:
        background.add_task(lambda: requests.post(webhook_url, json=result, timeout=10))

    return result


@router.post("/test/webhook-receiver")
def webhook_receiver(payload: Dict[str, Any]):
    logger.info("webhook_receive", extra={"payload_keys": list(payload.keys())})
    return {"received": True}


@router.get("/files")
def get_file(path: str, _=Depends(verify_jwt_token)):
    # Get the correct base directory (parent of api_app)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    abs_path = os.path.join(base_dir, path)
    
    # Debug logging
    logger.info(f"File request - path: {path}, base_dir: {base_dir}, abs_path: {abs_path}")
    logger.info(f"Path exists: {os.path.exists(abs_path)}")
    
    # Security check: ensure the resolved path is within the base directory
    if not abs_path.startswith(base_dir):
        logger.error(f"Access denied - path outside base directory: {abs_path}")
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.exists(abs_path):
        logger.error(f"File not found: {abs_path}")
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(abs_path)


app.include_router(router)


