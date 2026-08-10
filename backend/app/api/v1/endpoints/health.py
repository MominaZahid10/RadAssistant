"""
RadAssist AI — Health Check Endpoint

WHY A HEALTH CHECK?
Every production system needs a way to answer: "Are you alive?"
This endpoint checks not just "is the server running?" but also
"can it reach the database?" and "can it reach Qdrant?"

WHO CALLS THIS?
- Docker Compose uses it to know if the backend is ready
- Monitoring tools call it periodically to detect outages
- The frontend can call it to show connection status
- YOU during development to verify everything is connected

HTTP STATUS CODES:
- 200 = Everything is fine
- 503 = Service Unavailable (something is broken)
"""

from fastapi import APIRouter, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config import get_settings
from app.core.database import engine
from app.services.embedding import embedding_service
from app.services.qdrant_service import qdrant_service
from app.services.llm_service import llm_service

router = APIRouter()
settings = get_settings()


@router.get(
    "/health",
    summary="System Health Check",
    description="Verifies that the API, database, and vector store are operational.",
    tags=["System"],
)
async def health_check(verify_llm: bool = False):
    """
    Check the health of all system components.

    Returns a JSON object with the status of:
    - api:             Always "healthy" if this code runs
    - database:        "connected" or "disconnected"
    - qdrant:          "connected" or "disconnected"
    - embedding_model: loaded model name and dimension
    - llm:             active provider/model, and which keys are present
    - overall status:  "healthy" only if ALL components are up

    Query params:
        verify_llm — if true, makes a real (tiny) LLM API call to confirm the
                     key works. Off by default so that monitoring polling this
                     endpoint every few seconds doesn't burn tokens or hit rate
                     limits.
    """
    health = {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "components": {
            "api": "healthy",
            "database": "checking...",
            "qdrant": "checking...",
            "embedding_model": "checking...",
            "llm": "checking...",
        }
    }

    # ── Check PostgreSQL ─────────────────────────────────────
    try:
        # Execute a simple query to verify the connection works
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        health["components"]["database"] = "connected"
    except Exception as e:
        health["components"]["database"] = f"disconnected: {str(e)}"
        health["status"] = "degraded"

    # ── Check Qdrant ─────────────────────────────────────────
    # Reuse the shared client rather than building a new one per request —
    # constructing a QdrantClient opens a fresh connection pool each time,
    # which is wasteful for an endpoint that monitoring hits every few seconds.
    #
    # The client is synchronous, so it goes through the threadpool to avoid
    # blocking the event loop if Qdrant is slow to answer.
    try:
        await run_in_threadpool(qdrant_service.client.get_collections)
        health["components"]["qdrant"] = "connected"
    except Exception as e:
        health["components"]["qdrant"] = f"disconnected: {str(e)}"
        health["status"] = "degraded"

    # ── Check the embedding model ────────────────────────────
    # Without this, ingestion and search fail while /health still says
    # everything is fine — exactly the blind spot a health check should close.
    if embedding_service.is_loaded:
        info = embedding_service.get_info()
        health["components"]["embedding_model"] = (
            f"loaded: {info['model_name']} ({info['dimension']}D)"
        )
    else:
        health["components"]["embedding_model"] = "not loaded"
        health["status"] = "degraded"

    # ── Check the LLM provider (Phase 3) ─────────────────────
    # Config-only by default: reports whether a key exists, without spending
    # tokens. Pass ?verify_llm=true to make a real 5-token API call.
    llm_info = llm_service.get_provider_info()
    configured = [n for n, p in llm_info["providers"].items() if p["configured"]]

    if not configured:
        health["components"]["llm"] = "not configured — /chat will return 503"
        health["status"] = "degraded"
    elif verify_llm:
        check = await llm_service.check_connectivity()
        if check["status"] == "ok":
            suffix = "" if check.get("is_primary", True) else " (via fallback)"
            health["components"]["llm"] = (
                f"verified: {check['provider']} / {check['model']}{suffix}"
            )
        else:
            health["components"]["llm"] = f"unreachable: {check.get('error', 'unknown')}"
            health["status"] = "degraded"
    else:
        health["components"]["llm"] = (
            f"configured: {llm_info['active_provider']} / "
            f"{llm_info['active_model']} "
            f"(keys: {', '.join(configured)})"
        )

    # ── Return appropriate HTTP status ────────────────────────
    if health["status"] != "healthy":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health,
        )

    return health
