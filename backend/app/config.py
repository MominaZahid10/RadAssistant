"""
RadAssist AI — Application Configuration

WHY THIS FILE EXISTS:
Instead of hardcoding database URLs, API keys, and other settings
directly in our code (bad practice & security risk), we read them
from environment variables. This file defines WHAT variables we
expect and provides sensible defaults for local development.

HOW IT WORKS:
- Pydantic's BaseSettings automatically reads from a .env file
- If a variable has no default and isn't in .env, the app crashes
  at startup with a clear error (fail-fast principle)
- Type validation happens automatically (e.g., int stays int)
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    All configuration for RadAssist AI.
    
    Each field maps to an environment variable. For example:
        APP_NAME  →  settings.APP_NAME
        DATABASE_URL  →  settings.DATABASE_URL
    """

    # ── App Info ──────────────────────────────────────────────
    APP_NAME: str = "RadAssist AI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # ── Database (PostgreSQL) ─────────────────────────────────
    # For local dev: runs in Docker. For production: Supabase URL.
    DATABASE_URL: str = "postgresql+asyncpg://radassist:radassist_secret@postgres:5432/radassist_db"

    # ── Vector Database (Qdrant) ──────────────────────────────
    # Qdrant stores document embeddings for semantic search.
    # This is the core of our RAG retrieval system.
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333

    # ── LLM Configuration ────────────────────────────────────
    # We use Mistral for dev/testing, OpenAI for final testing.
    MISTRAL_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    
    # Which LLM provider to use: "mistral" or "openai"
    LLM_PROVIDER: str = "mistral"

    # ── CORS (Cross-Origin Resource Sharing) ──────────────────
    # Allows the frontend (port 3000) to call the backend (port 8000).
    # Without this, the browser blocks the requests for security.
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    class Config:
        # Tell Pydantic to read from this file
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Make field names case-insensitive
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    
    WHY @lru_cache?
    Reading .env and creating Settings is slow-ish. We only need to
    do it once — after that, every call returns the same instance.
    This is a common Python pattern called "singleton via cache."
    """
    return Settings()
