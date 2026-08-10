# ══════════════════════════════════════════════════════════════
# Pydantic Schemas Package
# ══════════════════════════════════════════════════════════════
# Schemas define the SHAPE of API requests and responses.
# Unlike models (which map to DB tables), schemas map to JSON.
#
# Import all schemas here so endpoints can do:
#   from app.schemas import DocumentResponse, SearchRequest
# instead of:
#   from app.schemas.document import DocumentResponse

from app.schemas.document import (  # noqa: F401
    DocumentResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    ChunkPreview,
    ChunkListResponse,
    SearchRequest,
    SearchResult,
    SearchResponse,
    KnowledgeBaseStats,
)

from app.schemas.rag import (  # noqa: F401  — Phase 3
    Audience,
    ChatRequest,
    ChatResponse,
    SourceReference,
    ProviderInfo,
    ModelInfoResponse,
)
