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
from app.core.database import engine, Base
from app.models import Document  # noqa: F401 — Import so SQLAlchemy discovers it
from app.services.embedding import embedding_service
from app.services.qdrant_service import qdrant_service

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
    # Create database tables from our models (including new 'documents' table)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database tables created/verified")
    
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
        print(f"⚠️  Embedding model failed to load: {e}")
        print("   Knowledge base ingestion and search will not work.")
    
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
