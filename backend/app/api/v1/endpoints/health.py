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

router = APIRouter()
settings = get_settings()


@router.get(
    "/health",
    summary="System Health Check",
    description="Verifies that the API, database, and vector store are operational.",
    tags=["System"],
)
async def health_check():
    """
    Check the health of all system components.
    
    Returns a JSON object with the status of:
    - api: Always "healthy" if this code runs
    - database: "connected" or "disconnected" 
    - qdrant: "connected" or "disconnected"
    - overall status: "healthy" only if ALL components are up
    """
    health = {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "components": {
            "api": "healthy",
            "database": "checking...",
            "qdrant": "checking...",
            "embedding_model": "checking...",
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

    # ── Return appropriate HTTP status ────────────────────────
    if health["status"] != "healthy":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health,
        )

    return health
