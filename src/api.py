"""
FastAPI backend for the SHL Assessment Recommendation System.

I built this API around two main concerns:
  1. It needs to work seamlessly with SHL's automated evaluation pipeline,
     so I intentionally kept CORS open and didn't add JWT/OAuth (would block
     their evaluator). I documented what I'd add in production below.
  2. Security still matters - I added rate limiting, input sanitization,
     SSRF protection, security headers, and error masking so the API
     handles adversarial input gracefully without leaking internals.

Endpoints:
    GET  /health          -> Simple liveness check
    POST /recommend       -> Main recommendation endpoint (JSON body)
    GET  /recommend       -> Same thing but query as URL param (for browsers)

Production improvements I'd add:
    - JWT or API key authentication (with SHL evaluator whitelisted)
    - Strict CORS whitelist (frontend domain + SHL eval domain only)
    - Request logging to persistent store (ELK/CloudWatch)
    - Redis-backed rate limiting instead of in-memory

Author: Mohammad Inayat Hussain
"""

import os
import sys
import time
import uuid
import logging
import traceback
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

load_dotenv()

# Make sure we can import from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import sanitize_input
# NOTE: AssessmentEngine is imported lazily in _load_engine() to avoid
# blocking the server startup. Importing sentence_transformers + PyTorch
# takes 3+ minutes on Render's free tier, which prevents port binding.


# -- Config (all overridable via .env) ----------------------------------------

RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
MAX_BODY_SIZE = int(os.getenv("MAX_REQUEST_SIZE_BYTES", "1048576"))  # 1 MB
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_QUERY_LEN = 10_000

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shl_api")


# -- Rate limiter -------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)


# -- Engine lifecycle ----------------------------------------------------------
# Load engine in a background thread so the API port opens immediately.
# This is critical for Render's free tier, which needs to detect the port
# within ~5 minutes. Model loading can take longer than that.

engine_instance: AssessmentEngine | None = None
engine_loading = True


def _load_engine():
    """Background loader for the recommendation engine."""
    global engine_instance, engine_loading
    try:
        from src.engine import AssessmentEngine  # Lazy import — heavy deps
        engine_instance = AssessmentEngine()
        log.info(f"Engine ready: {len(engine_instance.df)} assessments loaded")
    except Exception as e:
        log.error(f"CRITICAL: Engine failed to load: {e}")
        import traceback
        traceback.print_exc()
        engine_instance = None
    finally:
        engine_loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine_loading
    log.info("Starting SHL SmartMatch AI API...")
    import threading
    loader = threading.Thread(target=_load_engine, daemon=True)
    loader.start()
    log.info("Engine loading in background — API is accepting requests")
    yield
    log.info("Shutting down API...")


# -- App creation --------------------------------------------------------------

app = FastAPI(
    title="SHL SmartMatch AI",
    description=(
        "AI-powered assessment recommender. Accepts natural language queries "
        "or job description URLs and returns the most relevant SHL assessments."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# -- Middleware stack -----------------------------------------------------------
# Order matters here. FastAPI processes middleware in reverse registration order,
# so the last one registered runs first. I'm registering in logical order:
# CORS -> Security Headers -> Body size check -> Request logging -> Error handler

# CORS - wide open for SHL's automated evaluator to call us directly.
# In production I'd whitelist just our frontend + SHL's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Inject standard security headers into every response."""
    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Let the client know our rate limit policy
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT)
    response.headers["X-RateLimit-Window"] = "1 minute"

    # Swagger UI needs CDN access, so relax CSP for docs pages only
    if request.url.path in ("/docs", "/redoc", "/openapi.json"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https://fastapi.tiangolo.com"
        )
    else:
        response.headers["Content-Security-Policy"] = "default-src 'self'"

    return response


@app.middleware("http")
async def check_body_size(request: Request, call_next):
    """Reject oversized payloads early to prevent memory exhaustion."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_SIZE:
        return JSONResponse(
            status_code=413,
            content={"error": "Payload too large", "max_bytes": MAX_BODY_SIZE},
        )
    return await call_next(request)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    """
    Log request metadata for observability.
    I'm deliberately NOT logging the full query text - it could contain
    sensitive JD data. Just logging the method, path, and timing.
    """
    t0 = time.time()
    client_ip = request.client.host if request.client else "unknown"

    response = await call_next(request)

    ms = (time.time() - t0) * 1000
    log.info(f"REQ {request.method} {request.url.path} | IP={client_ip} | "
             f"status={response.status_code} | {ms:.0f}ms")
    return response


@app.exception_handler(Exception)
async def catch_all_errors(request: Request, exc: Exception):
    """
    Global safety net. If anything unexpected blows up, the client gets
    a clean 500 with an error_id they can report. The full traceback only
    goes to our server logs, never to the response body.
    """
    error_id = str(uuid.uuid4())[:8]
    log.error(f"UNHANDLED [{error_id}]: {type(exc).__name__}: {exc}\n"
              f"{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "error_id": error_id},
    )


# -- Request/Response schemas --------------------------------------------------

class RecommendRequest(BaseModel):
    """JSON body for POST /recommend."""
    query: str = Field(
        ...,
        description="Natural language query or job description URL",
        min_length=1,
        max_length=MAX_QUERY_LEN,
        examples=["I need a cognitive ability test for entry-level candidates under 30 minutes"],
    )


class AssessmentResult(BaseModel):
    """One recommended assessment in the response."""
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: list[str]


class RecommendResponse(BaseModel):
    """Wire format for /recommend responses - matches SHL's spec exactly."""
    recommended_assessments: list[AssessmentResult]


# -- Shared recommendation logic -----------------------------------------------

def _get_recommendations(query: str) -> list[dict]:
    """
    Core logic shared by both GET and POST recommend endpoints.
    Sanitizes input, calls the engine, and enriches the response
    with descriptions from the catalog.
    """
    if engine_instance is None:
        if engine_loading:
            raise HTTPException(503, "Engine is loading, please retry in 30 seconds")
        raise HTTPException(503, "Service temporarily unavailable - engine not loaded")

    clean_query = sanitize_input(query)
    if not clean_query or not clean_query.strip():
        raise HTTPException(400, "Query is empty after sanitization. Please provide a valid query.")

    results = engine_instance.recommend(clean_query, top_k=10)

    # Enrich each result with the full description from the catalog
    response_items = []
    for r in results:
        desc = ""
        if engine_instance.df is not None:
            match = engine_instance.df[engine_instance.df["url"] == r["url"]]
            if len(match) > 0:
                desc = str(match.iloc[0].get("description", ""))

        response_items.append({
            "url": r["url"],
            "adaptive_support": r.get("adaptive_support", "No"),
            "description": desc[:500],
            "duration": r.get("duration", -1),
            "remote_support": r.get("remote_support", "Yes"),
            "test_type": r.get("test_type", []),
        })

    return response_items


# -- Endpoints -----------------------------------------------------------------

@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    """Root endpoint — shows a welcome message so / doesn't return 404."""
    return {
        "service": "SHL SmartMatch AI",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "recommend_post": "POST /recommend",
            "recommend_get": "GET /recommend?query=<text>",
        },
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Quick liveness probe. Also reports how many assessments are loaded."""
    if engine_loading:
        return {"status": "healthy", "engine": "loading", "assessments": 0}
    return {
        "status": "healthy",
        "engine": "loaded" if engine_instance else "error",
        "assessments": len(engine_instance.df) if engine_instance else 0,
    }


@app.post(
    "/recommend",
    response_model=RecommendResponse,
    tags=["Recommendations"],
    summary="Get assessment recommendations (POST)",
    description="Submit a natural language query or JD URL to get the top 10 matching SHL assessments.",
)
@limiter.limit(f"{RATE_LIMIT}/minute")
async def recommend_post(request: Request, body: RecommendRequest):
    """
    POST /recommend - primary endpoint for the SHL evaluation pipeline.
    Accepts JSON body with a 'query' field.
    """
    results = _get_recommendations(body.query)
    return {"recommended_assessments": results}


@app.get(
    "/recommend",
    response_model=RecommendResponse,
    tags=["Recommendations"],
    summary="Get assessment recommendations (GET)",
    description="Pass query as a URL parameter. Handy for quick browser testing.",
)
@limiter.limit(f"{RATE_LIMIT}/minute")
async def recommend_get(
    request: Request,
    query: str = Query(
        ...,
        description="Natural language query or job description URL",
        min_length=1,
        max_length=MAX_QUERY_LEN,
        examples=["Java developer coding test under 30 minutes"],
    ),
):
    """GET /recommend?query=<text> - convenience endpoint, same logic as POST."""
    results = _get_recommendations(query)
    return {"recommended_assessments": results}
