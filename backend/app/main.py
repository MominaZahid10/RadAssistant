"""
RadAssist AI — FastAPI Application Entry Point

THIS IS THE HEART OF THE BACKEND.

When you run the backend, this is the first file that executes.
It does three things:

1. CREATES the FastAPI app with metadata (title, description, version)
2. ADDS middleware (CORS, logging, etc.)
3. REGISTERS all API routes

WHAT IS MIDDLEWARE?
Middleware is code that runs BEFORE and AFTER every request.
Think of it as a security checkpoint at an airport:
- Every passenger (request) must pass through it
- It can modify, block, or inspect requests/responses

WHAT IS CORS?
Cross-Origin Resource Sharing. Browsers block requests from
one domain (localhost:3000) to another (localhost:8000) by default.
CORS middleware tells the browser "it's OK, let the frontend
talk to me."

WHAT ARE LIFESPAN EVENTS?
Code that runs once when the server STARTS and once when it STOPS.
We use startup to create database tables and verify connections.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.config import get_settings
from app.api.v1.router import api_v1_router
from app.core import errors
from app.core.database import engine
from app.models import Document  # noqa: F401 — Import so SQLAlchemy discovers it
from app.services.embedding import embedding_service
from app.services.qdrant_service import qdrant_service
from app.services.llm_service import llm_service

settings = get_settings()


# ── Lifespan: Startup & Shutdown ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup (before 'yield') and once at shutdown (after 'yield').
    
    STARTUP (Phase 1):
    - Creates all database tables if they don't exist yet
    - Verifies database connectivity
    
    STARTUP (Phase 2 — NEW):
    - Loads the embedding model into memory (2-5 seconds)
    - Creates/verifies the Qdrant vector collection
    - Creates the upload directory for temporary file storage
    
    SHUTDOWN:
    - Closes all database connections cleanly
    """
    # ── STARTUP ──
    print(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print("="  * 50)
    
    # ── Phase 1: Database ────────────────────────────────────
    # ⚠️  create_all() WAS REMOVED HERE IN PHASE 4 STEP 0.
    #
    # It creates tables but never ALTERS them. Add a column to a model and
    # create_all does nothing — silently. No error, no change; the app then
    # fails at query time with a missing-column error that points nowhere
    # near the cause. That's the same class of silent failure that has bitten
    # this project repeatedly.
    #
    # Schema changes now go through Alembic:
    #     docker-compose exec backend alembic revision --autogenerate -m "..."
    #     docker-compose exec backend alembic upgrade head
    #
    # Migrations are applied by the container's start command, before uvicorn
    # boots, so the app never runs against a schema it wasn't built for. This
    # block only VERIFIES that they ran.
    try:
        async with engine.connect() as conn:
            version = await conn.scalar(
                text("SELECT version_num FROM alembic_version LIMIT 1")
            )
        if version:
            print(f"✅ Database schema at migration: {version}")
        else:
            print("⚠️  alembic_version is empty — run: alembic upgrade head")
    except Exception:
        print("=" * 50)
        print("⚠️  No alembic_version table — migrations have not been applied.")
        print("")
        print("   The app may be running against a schema it wasn't built for.")
        print("   Apply them with:")
        print("       docker-compose exec backend alembic upgrade head")
        print("=" * 50)
    
    # Verify database connection
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        print("✅ PostgreSQL connection verified")
    except Exception as e:
        print(f"⚠️  PostgreSQL connection failed: {e}")
    
    # ── Phase 2: Embedding Model ─────────────────────────────
    # Load the AI model that converts text into vectors.
    # This takes 2-5 seconds on first run (downloads ~80MB model).
    # After that, it loads from cache in <1 second.
    try:
        embedding_service.load()
        model_info = embedding_service.get_info()
        print(f"✅ Embedding model ready: {model_info['model_name']} ({model_info['dimension']}D)")
    except Exception as e:
        print("=" * 50)
        print(f"❌ EMBEDDING MODEL FAILED TO LOAD: {e}")
        print("")
        print("   ALL ingestion and search will fail until this is fixed.")
        print("   Documents will be accepted and then marked 'failed'.")
        print("")
        print("   Most likely cause: the model cache was wiped (e.g. by")
        print("   `docker-compose down -v`) while HF_HUB_OFFLINE=1 prevents")
        print("   re-downloading it.")
        print("")
        print("   Fix:  HF_HUB_OFFLINE=0 docker-compose up -d backend")
        print("=" * 50)
    
    # ── Phase 2: Qdrant Collection ───────────────────────────
    # Create the vector collection if it doesn't exist.
    # This is idempotent — safe to call every startup.
    try:
        qdrant_service.ensure_collection()
    except Exception as e:
        print(f"⚠️  Qdrant collection setup failed: {e}")
        print("   Vector storage and search will not work.")
    
    # ── Phase 2: Upload Directory ────────────────────────────
    # Create the directory where uploaded files are temporarily saved.
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    print(f"✅ Upload directory ready: {settings.UPLOAD_DIR}")

    # ── Phase 4: Image storage ───────────────────────────────
    os.makedirs(settings.IMAGE_DIR, exist_ok=True)
    print(f"✅ Image directory ready: {settings.IMAGE_DIR}")

    # ── Phase 3: LLM Provider ────────────────────────────────
    # Report which providers have keys. We deliberately do NOT make a live
    # API call here: it would add latency to every container start, cost
    # tokens on each restart, and fight with Docker healthchecks. The real
    # connectivity test lives behind GET /api/v1/health, which you can call
    # on demand.
    info = llm_service.get_provider_info()
    configured = [n for n, p in info["providers"].items() if p["configured"]]

    if configured:
        print(
            f"✅ LLM ready: {info['active_provider']} / {info['active_model']}"
        )
        fallbacks = [n for n in configured if n != info["active_provider"]]
        if fallbacks:
            print(f"   Fallback providers available: {', '.join(fallbacks)}")
        if info["active_provider"] not in configured:
            print(
                f"⚠️  LLM_PROVIDER is '{info['active_provider']}' but it has no "
                f"API key — requests will fall back to: {', '.join(configured)}"
            )
    else:
        print("⚠️  No LLM provider configured — /api/v1/chat will return 503")
        print("   Set GROQ_API_KEY, MISTRAL_API_KEY, or OPENAI_API_KEY in .env")

    print("=" * 50)
    print(f"🟢 {settings.APP_NAME} is ready!")
    print(f"   Docs:   http://localhost:8000/docs")
    print(f"   Health: http://localhost:8000/api/v1/health")
    print("=" * 50)
    
    yield  # ← App runs here, serving requests
    
    # ── SHUTDOWN ──
    print("🛑 Shutting down...")
    await engine.dispose()
    print("✅ Database connections closed")


# ── Create the FastAPI App ───────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "An Explainable, Multimodal Retrieval-Augmented Radiology "
        "Reporting & Clinical Decision Support System"
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    # Swagger UI will be available at /docs
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── CORS Middleware ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # Which domains can call us
    allow_credentials=True,                # Allow cookies/auth headers
    allow_methods=["*"],                   # Allow all HTTP methods
    allow_headers=["*"],                   # Allow all headers
)


# ── Register API Routes ─────────────────────────────────────
# ⚠️  INSTALLED BEFORE THE ROUTES ARE MOUNTED.
# Gives every request a correlation id and stops exception text reaching the
# caller. Endpoints previously did `detail=f"Internal error: {e}"`, which
# hands out file paths and driver messages. See app/core/errors.py.
errors.install(app)

app.include_router(api_v1_router)


# ── Root Endpoint ────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint — confirms the API is running.
    Useful for quick "is this alive?" checks.
    """
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
