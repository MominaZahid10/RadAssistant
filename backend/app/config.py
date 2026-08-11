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

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


def _split_csv(raw: str) -> list[str]:
    """
    Parse a comma-separated (or JSON-array) env var into a list of strings.

    Accepts both forms so neither is a footgun:
        CORS_ORIGINS=http://a.com,http://b.com
        CORS_ORIGINS=["http://a.com","http://b.com"]
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        import json
        try:
            return [str(x).strip() for x in json.loads(raw)]
        except (ValueError, TypeError):
            pass  # Malformed JSON — fall through to comma parsing
    return [item.strip() for item in raw.split(",") if item.strip()]


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

    # ── LLM Configuration (Phase 3) ─────────────────────────
    # Three providers, switchable via LLM_PROVIDER:
    #   groq    — Free tier, fast inference (development default)
    #   mistral — Free tier (limited), quality models (fallback)
    #   openai  — Paid, highest quality (production)
    #
    # The service tries the primary provider first, then falls back
    # through the others if the primary fails (rate limit, missing key).
    GROQ_API_KEY: str = ""
    MISTRAL_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Which LLM provider to use: "groq", "mistral", or "openai"
    LLM_PROVIDER: str = "groq"

    # Model override — leave blank to auto-select per provider:
    #   groq    → llama-3.3-70b-versatile
    #   mistral → mistral-large-latest
    #   openai  → gpt-4o-mini
    LLM_MODEL: str = ""

    # Temperature: 0.0–1.0.  Low = deterministic, factual.
    # Clinical text shouldn't be creative, and low variance makes
    # RAG evaluation (recall@5) reproducible across runs.
    LLM_TEMPERATURE: float = 0.2

    # Max tokens in the LLM response.  2048 ≈ ~1500 words — generous
    # for a Q&A answer, tight enough to prevent runaway generation.
    LLM_MAX_TOKENS: int = 2048

    # ── Embedding Model (Phase 2) ────────────────────────────
    # The embedding model converts text into vectors (lists of numbers).
    # These vectors capture the MEANING of text, so similar medical
    # concepts end up close together in vector space.
    #
    # all-MiniLM-L6-v2: Fast, lightweight (80MB), 384 dimensions.
    # Good for prototyping. Can swap to PubMedBERT (768-dim) later
    # for better medical accuracy — just change these two values.
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32  # How many chunks to embed at once

    # ── Reranking (Phase 3.5) ────────────────────────────────
    # Two-stage retrieval: vector search finds candidates (recall), then a
    # cross-encoder rescores them by actual query-passage relevance
    # (precision). Measured need: source recall@12 was 88.9% but keyword
    # recall@12 only 72.2% — we were finding the right documents and
    # retrieving the wrong passages from them.
    #
    # Set RERANK_ENABLED=false to fall back to pure vector ordering. The
    # service also degrades to that automatically if the model can't load,
    # so a failed download never breaks chat.
    RERANK_ENABLED: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # How many vector hits to rescore. Bigger = better recall before the
    # precision stage, at ~1ms per candidate on CPU. 48 ≈ 50ms, which is
    # comparable to the vector search itself.
    RERANK_CANDIDATES: int = 48

    # Chunks retrieved per question, and the most any single document may
    # contribute. Both are tuning knobs — exposed as env vars so they can be
    # swept against the evaluation harness without editing code:
    #
    #   RETRIEVAL_LIMIT=12 MAX_CHUNKS_PER_DOCUMENT=5 docker-compose up -d backend
    #   cd backend && python eval/run_eval.py --vs vector_only
    #
    # MEASURED TENSION: capping at 3 forces source diversity (good for an
    # evidence panel — corroboration beats four excerpts from one paper), but
    # three evaluation questions retrieve the correct document at rank 1 and
    # STILL miss the answering passage, because it's that document's 4th-best
    # chunk. Diversity and depth trade directly against each other here.
    # ── Hybrid retrieval (Phase 3.6) ─────────────────────────
    # BM25 lexical search unioned with vector search before reranking.
    # Measured need: the chunk answering "radiographic findings of
    # pneumothorax" exists in the corpus and never entered the vector
    # candidate pool, so the cross-encoder could not reach it. Embeddings
    # find passages ABOUT a topic; BM25 finds passages CONTAINING the
    # exact terms. Their failure modes are near-orthogonal.
    HYBRID_ENABLED: bool = True
    LEXICAL_CANDIDATES: int = 20

    RETRIEVAL_LIMIT: int = 12
    MAX_CHUNKS_PER_DOCUMENT: int = 3

    # ── Text Chunking (Phase 2) ──────────────────────────────
    # Documents are split into small "chunks" before embedding.
    # WHY? Embedding models have a token limit (~256 tokens for MiniLM).
    # Smaller chunks also give more precise search results.
    #
    # CHUNK_SIZE: Max characters per chunk (~512 chars ≈ 100-130 words)
    # CHUNK_OVERLAP: Characters shared between consecutive chunks.
    #   This prevents losing context at chunk boundaries. For example,
    #   if a finding spans two chunks, the overlap ensures it appears
    #   fully in at least one of them.
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # ── Qdrant Collection (Phase 2) ──────────────────────────
    # A "collection" in Qdrant is like a table in PostgreSQL —
    # it holds all vectors for a specific purpose.
    # We use one collection for all medical knowledge.
    QDRANT_COLLECTION: str = "radassist_knowledge"

    # ── File Upload Settings (Phase 2) ───────────────────────
    # Controls what files doctors can upload and how big they can be.
    MAX_UPLOAD_SIZE_MB: int = 50

    # ⚠️  DECLARED AS str, NOT list[str] — ON PURPOSE.
    # pydantic-settings runs json.loads() on any "complex" field type
    # (list/dict/set) inside its *env source*, which happens BEFORE any
    # field_validator gets a chance to run. So a plain value like
    #     ALLOWED_EXTENSIONS=pdf,docx
    # crashes the app at startup with a JSONDecodeError, and no validator
    # can rescue it. Keeping the field a str sidesteps the decoder entirely;
    # the parsed list is exposed via the property below.
    ALLOWED_EXTENSIONS_RAW: str = Field(
        default=".pdf,.docx,.txt,.md,.png,.jpg,.jpeg,.tiff,.bmp",
        alias="ALLOWED_EXTENSIONS",
    )
    # Where uploaded files are temporarily saved during processing
    UPLOAD_DIR: str = "/app/uploads"

    # ── NCBI / PubMed API (Phase 2) ──────────────────────────
    # Used to fetch medical articles from StatPearls and PubMed.
    # StatPearls is a peer-reviewed medical knowledge base on NCBI —
    # freely accessible, regularly updated, used by clinicians worldwide.
    #
    # Get a free API key at: https://www.ncbi.nlm.nih.gov/account/
    # Without a key: 3 requests/sec. With key: 10 requests/sec.
    NCBI_API_KEY: str = ""
    NCBI_EMAIL: str = ""  # NCBI requires an email for API access

    # ── CORS (Cross-Origin Resource Sharing) ──────────────────
    # Allows the frontend (port 3000) to call the backend (port 8000).
    # Without this, the browser blocks the requests for security.
    # Same str-not-list treatment as ALLOWED_EXTENSIONS above.
    CORS_ORIGINS_RAW: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="CORS_ORIGINS",
    )

    # ── Parsed views ─────────────────────────────────────────
    # These keep every call site unchanged: code still reads
    # settings.CORS_ORIGINS and settings.ALLOWED_EXTENSIONS and gets a list.

    @property
    def CORS_ORIGINS(self) -> list[str]:
        """Origins allowed to call this API, parsed from the raw env value."""
        return _split_csv(self.CORS_ORIGINS_RAW)

    @property
    def ALLOWED_EXTENSIONS(self) -> list[str]:
        """
        Uploadable file extensions, normalised to lowercase with a leading dot.
        Accepts 'pdf', '.pdf', or 'PDF' in config and yields '.pdf'.
        """
        return [
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in _split_csv(self.ALLOWED_EXTENSIONS_RAW)
        ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Allow setting fields by their alias (ALLOWED_EXTENSIONS) as well as
        # by their Python name (ALLOWED_EXTENSIONS_RAW).
        populate_by_name=True,
        # Ignore unrelated variables that Docker/the OS injects into the
        # environment, instead of crashing on them.
        extra="ignore",
    )


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
