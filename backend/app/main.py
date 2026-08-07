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

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.config import get_settings
from app.api.v1.router import api_v1_router
from app.core.database import engine, Base

settings = get_settings()


# ── Lifespan: Startup & Shutdown ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup (before 'yield') and once at shutdown (after 'yield').
    
    STARTUP:
    - Creates all database tables if they don't exist yet
    - Verifies database connectivity
    
    SHUTDOWN:
    - Closes all database connections cleanly
    """
    # ── STARTUP ──
    print(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Create database tables from our models
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
