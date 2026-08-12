"""
RadAssist AI — Document Schemas (API Request/Response Shapes)

WHAT ARE SCHEMAS?
Schemas define the EXACT shape of data going IN and OUT of our API.
They are NOT database tables — they are JSON contracts.

Think of it like a form:
- A schema for REQUESTS says: "You must send me these fields"
- A schema for RESPONSES says: "I will send you back these fields"

WHY SEPARATE FROM MODELS?
The database model (models/document.py) has EVERY column, including
internal ones like error_message. We don't always want to expose
everything to the frontend. Schemas let us control exactly what
data goes in and out.

For example:
- When uploading, the frontend sends: file + source_type + title
- When listing, the API returns: id, filename, status, chunk_count
  (but NOT the raw error_message or internal processing details)

WHAT IS Pydantic?
A Python library that validates data automatically. If the frontend
sends a string where we expect an integer, Pydantic raises a clear
error BEFORE our code even runs. FastAPI uses Pydantic schemas to:
1. Validate incoming request data
2. Generate the Swagger/OpenAPI documentation automatically
3. Serialize Python objects to JSON responses
"""

from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


# ══════════════════════════════════════════════════════════════
# RESPONSE SCHEMAS — What the API sends BACK to the frontend
# ══════════════════════════════════════════════════════════════


class DocumentResponse(BaseModel):
    """
    Standard response when returning a single document's info.
    Used after upload, when checking status, or viewing details.
    
    The frontend uses this to show document cards in the dashboard:
    ┌──────────────────────────────────────────────┐
    │ 📄 chest_xray_guidelines.pdf                 │
    │ Type: guideline  |  Status: ✅ completed     │
    │ 45 chunks  |  Uploaded: 2024-08-08           │
    └──────────────────────────────────────────────┘
    """
    id: UUID
    filename: str
    file_type: str
    file_size: int | None = None
    title: str | None = None
    source_type: str
    source_url: str | None = None
    description: str | None = None
    status: str
    chunk_count: int
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    """
    Paginated list of documents.
    
    WHY PAGINATION?
    If the knowledge base has 1000 documents, we don't want to send
    ALL of them at once — that's slow and wastes bandwidth. Instead,
    we send 20 at a time and let the frontend request more.
    
    Example response:
    {
        "documents": [...20 items...],
        "total": 156,
        "page": 1,
        "page_size": 20
    }
    """
    documents: list[DocumentResponse]
    total: int
    page: int
    page_size: int


class DocumentUploadResponse(BaseModel):
    """
    Response immediately after a successful upload.
    Tells the frontend the document ID and its current status.
    
    The frontend can then poll the status endpoint to check
    when processing is complete (or use WebSockets in future).
    """
    id: UUID
    filename: str
    status: str
    message: str


# ══════════════════════════════════════════════════════════════
# CHUNK SCHEMAS — Preview what the ingestion pipeline produced
# ══════════════════════════════════════════════════════════════


class ChunkPreview(BaseModel):
    """
    A single chunk from a processed document.
    
    Used by the developer/admin to inspect how documents were split.
    This is crucial for debugging — if the RAG gives bad answers,
    you want to check: "Were the chunks split properly?"
    
    Example:
    {
        "chunk_index": 0,
        "text": "Pneumonia is an infection that inflames the air sacs...",
        "char_count": 487
    }
    """
    chunk_index: int
    text: str
    char_count: int


class ChunkListResponse(BaseModel):
    """List of chunks for a document, with pagination."""
    document_id: UUID
    filename: str
    chunks: list[ChunkPreview]
    total_chunks: int
    page: int
    page_size: int


# ══════════════════════════════════════════════════════════════
# SEARCH SCHEMAS — For testing retrieval (developer tool)
# ══════════════════════════════════════════════════════════════


class SearchRequest(BaseModel):
    """
    A semantic search query against the knowledge base.
    
    HOW SEARCH WORKS:
    1. The query text is embedded into a vector (same model as ingestion)
    2. Qdrant finds the closest vectors (= most similar content)
    3. Returns the matching text chunks with similarity scores
    
    Example:
    {
        "query": "What are the findings of pneumothorax on chest X-ray?",
        "limit": 5,
        "source_type": "textbook"    ← optional filter
    }
    """
    query: str = Field(
        ...,  # Required field
        min_length=3,
        max_length=1000,
        description="The search query — a medical question or topic"
    )
    limit: int = Field(
        default=5,
        ge=1,    # Greater than or equal to 1
        le=20,   # Less than or equal to 20
        description="How many results to return (1-20)"
    )
    source_type: str | None = Field(
        default=None,
        description="Optional filter: only search within a specific source type"
    )


class SearchResult(BaseModel):
    """
    A single search result — a chunk that matched the query.
    
    The 'score' is how similar this chunk is to the query:
    - 1.0 = perfect match (identical meaning)
    - 0.8+ = very relevant
    - 0.5-0.8 = somewhat relevant
    - Below 0.5 = probably noise
    
    We return the source document info so the radiologist can
    trace WHERE the information came from (explainability).
    """
    text: str
    score: float
    document_id: str | None = None
    filename: str | None = None
    source_type: str | None = None
    chunk_index: int | None = None


class SearchResponse(BaseModel):
    """
    Complete search results with the original query echoed back.
    
    Example:
    {
        "query": "pneumothorax chest x-ray findings",
        "results": [...5 items...],
        "total_results": 5
    }
    """
    query: str
    results: list[SearchResult]
    total_results: int


# ══════════════════════════════════════════════════════════════
# KNOWLEDGE BASE STATS — Dashboard overview
# ══════════════════════════════════════════════════════════════


class KnowledgeBaseStats(BaseModel):
    """
    Overview statistics for the knowledge base.
    Shown on the admin dashboard to answer: "How healthy is our KB?"
    
    Example:
    {
        "total_documents": 42,
        "completed_documents": 40,
        "failed_documents": 2,
        "total_chunks": 3847,
        "source_type_counts": {
            "textbook": 5,
            "guideline": 12,
            "statpearls": 25
        }
    }
    """
    total_documents: int
    completed_documents: int
    failed_documents: int
    processing_documents: int
    total_chunks: int
    source_type_counts: dict[str, int]
    qdrant_vectors: int | None = None  # Total vectors in Qdrant
