"""
RadAssist AI — Knowledge Base API Endpoints

WHAT THIS FILE DOES:
Provides all the REST API endpoints for managing the knowledge base.
These are the URLs that the frontend (or Swagger UI) calls to:
- Upload documents (PDF, DOCX, TXT, images)
- List all ingested documents
- View document details and chunks
- Search the knowledge base semantically
- Delete documents
- View knowledge base statistics

HOW FastAPI ENDPOINTS WORK:
Each function decorated with @router.get/post/delete becomes an HTTP endpoint.
FastAPI automatically:
1. Validates request data using Pydantic schemas
2. Generates Swagger documentation
3. Returns proper HTTP status codes
4. Handles errors with clear messages

ENDPOINT OVERVIEW:
    POST   /api/v1/knowledge/upload                Upload a file (async)
    GET    /api/v1/knowledge/documents             List all documents
    GET    /api/v1/knowledge/documents/{id}        Get one document's details
    DELETE /api/v1/knowledge/documents/{id}        Delete a document
    GET    /api/v1/knowledge/documents/{id}/chunks Preview chunks
    POST   /api/v1/knowledge/search                Semantic search
    POST   /api/v1/knowledge/seed                  Load curated radiology content
    GET    /api/v1/knowledge/stats                 Knowledge base statistics
"""

import os
import uuid
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    BackgroundTasks,
    UploadFile,
    File,
    Form,
    HTTPException,
    status,
    Depends,
)
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.database import get_db, async_session
from app.models.document import Document
from app.schemas.document import (
    DocumentResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    ChunkListResponse,
    ChunkPreview,
    SearchRequest,
    SearchResponse,
    SearchResult,
    KnowledgeBaseStats,
)
from app.schemas.rag import PMCFetchRequest
from app.data.seed_knowledge import SEED_KNOWLEDGE
from app.services.ingestion import ingest_document
from app.services.embedding import embedding_service
from app.services.knowledge_seeder import seed_knowledge_base, ncbi_is_configured
from app.services.pmc_fetcher import fetch_and_ingest_pmc, DEFAULT_TOPICS
from app.services.qdrant_service import qdrant_service
from app.services.rag_service import rag_service

settings = get_settings()

# Create the router — all endpoints here get the /knowledge prefix
router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])


def _require_embedding_model() -> None:
    """
    Refuse ingestion outright when the embedding model isn't loaded.

    ⚠️  WHY THIS GUARD EXISTS:
    Without it, /seed and /fetch-pmc happily created 103 database rows, ran
    every one through the pipeline, and marked all 103 "failed" — producing a
    knowledge base that looked catastrophically broken when the real problem
    was a single missing model file. The user then has to clean up 103 rows
    before retrying.

    Checking once, up front, turns that into a clear 503 and zero side effects.
    """
    if not embedding_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The embedding model is not loaded, so nothing can be indexed. "
                "Ingestion is refused rather than creating documents that would "
                "all fail.\n\n"
                "Most likely the model cache is empty while HF_HUB_OFFLINE=1 "
                "blocks downloading it. Populate it once with a working "
                "connection:\n"
                "    HF_HUB_OFFLINE=0 docker-compose up -d backend\n\n"
                "Then set HF_HUB_OFFLINE=1 again. The cache lives in "
                "backend/.hf_cache on the host, so it survives docker-compose "
                "down -v."
            ),
        )


# ══════════════════════════════════════════════════════════════
# BACKGROUND WORKER
# ══════════════════════════════════════════════════════════════


async def _process_upload_in_background(
    file_bytes: bytes,
    filename: str,
    file_type: str,
    document_id: uuid.UUID,
    source_type: str,
    title: str | None,
) -> None:
    """
    Run the ingestion pipeline after the HTTP response has been sent.

    ⚠️  WHY ITS OWN DB SESSION?
    The session injected into the endpoint via Depends(get_db) is closed as
    soon as the response is returned. A background task that tried to reuse
    it would fail with "session is closed". So we open a fresh one here and
    own its full lifecycle.

    This function must never raise: an unhandled exception in a background
    task is swallowed by Starlette, which would leave the document stuck at
    status="processing" forever with no explanation. Everything is caught
    and written back to the row.
    """
    async with async_session() as db:
        try:
            result = await ingest_document(
                file_bytes=file_bytes,
                filename=filename,
                file_type=file_type,
                document_id=str(document_id),
                source_type=source_type,
                title=title,
            )
            new_status = result["status"]
            chunk_count = result["chunk_count"]
            error_message = result["message"] if new_status == "failed" else None
        except Exception as e:  # noqa: BLE001 — last line of defence
            new_status = "failed"
            chunk_count = 0
            error_message = f"Unhandled error during ingestion: {e}"

        # Write the outcome back so GET /documents/{id} reflects reality.
        try:
            doc = await db.get(Document, document_id)
            if doc is not None:
                doc.status = new_status
                doc.chunk_count = chunk_count
                doc.error_message = error_message
                doc.updated_at = datetime.now(timezone.utc)
                await db.commit()
        except Exception as e:  # noqa: BLE001
            await db.rollback()
            print(f"❌ Could not record ingestion result for {document_id}: {e}")


# ══════════════════════════════════════════════════════════════
# UPLOAD — Accept files from doctors
# ══════════════════════════════════════════════════════════════


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document to the knowledge base",
    description=(
        "Upload a PDF, DOCX, TXT, or image file. The file is parsed, chunked, "
        "embedded, and stored in the vector database for semantic search.\n\n"
        "This returns immediately with status='processing'. Poll "
        "`GET /knowledge/documents/{id}` until status becomes 'completed' or "
        "'failed'.\n\n"
        "Supported formats: PDF, DOCX, TXT, MD, PNG, JPG, JPEG, TIFF, BMP."
    ),
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(
        ...,
        description="The document file to upload"
    ),
    source_type: str = Form(
        default="general",
        description="Type of source: textbook, guideline, report, research_paper, clinical_note, general"
    ),
    title: str = Form(
        default=None,
        description="Optional title for the document (auto-extracted from filename if not provided)"
    ),
    description: str = Form(
        default=None,
        description="Optional description or notes about this document"
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document and queue it for ingestion.

    FLOW:
    1. Validate file type and size (synchronous — fail fast on bad input)
    2. Create a document record in PostgreSQL with status="processing"
    3. Return 201 immediately
    4. AFTER the response is sent, the ingestion pipeline runs in the
       background and updates the row to "completed" or "failed"

    WHY NOT DO THE WORK INLINE?
    Embedding a large PDF takes minutes of CPU. Holding the HTTP connection
    open that long causes client/proxy timeouts, and it makes the status
    column pointless — you'd only ever see the final state.

    The frontend polls GET /knowledge/documents/{id} to track progress.
    """
    _require_embedding_model()

    # ── Validate file extension ─────────────────────────────
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required."
        )
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File type '{file_ext}' is not supported. "
                f"Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        )
    
    # ── Read file bytes ─────────────────────────────────────
    file_bytes = await file.read()
    file_size = len(file_bytes)
    
    # ── Validate file size ──────────────────────────────────
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({file_size / 1024 / 1024:.1f}MB). Maximum: {settings.MAX_UPLOAD_SIZE_MB}MB."
        )
    
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty."
        )
    
    # ── Auto-generate title from filename if not provided ───
    if not title:
        title = os.path.splitext(file.filename)[0].replace("_", " ").replace("-", " ").title()
    
    # ── Create document record in PostgreSQL ────────────────
    doc = Document(
        filename=file.filename,
        file_type=file_ext.lstrip("."),
        file_size=file_size,
        title=title,
        source_type=source_type,
        description=description,
        status="processing",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # ── Queue the ingestion pipeline ────────────────────────
    # Starlette runs this after the response is sent.
    background_tasks.add_task(
        _process_upload_in_background,
        file_bytes=file_bytes,
        filename=file.filename,
        file_type=file_ext.lstrip("."),
        document_id=doc.id,
        source_type=source_type,
        title=title,
    )

    return DocumentUploadResponse(
        id=doc.id,
        filename=doc.filename,
        status=doc.status,  # "processing"
        message=(
            f"'{file.filename}' accepted and queued for processing. "
            f"Poll GET /api/v1/knowledge/documents/{doc.id} for status."
        ),
    )


# ══════════════════════════════════════════════════════════════
# LIST & GET — Browse the knowledge base
# ══════════════════════════════════════════════════════════════


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all documents in the knowledge base",
    description="Returns a paginated list of all ingested documents with their status and metadata.",
)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    source_type: str | None = None,
    status_filter: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    List all documents with optional filtering and pagination.
    
    Supports filtering by source_type ("textbook", "guideline", etc.)
    and by processing status ("completed", "failed", etc.).
    """
    # Build query with optional filters
    query = select(Document)
    count_query = select(func.count(Document.id))
    
    if source_type:
        query = query.where(Document.source_type == source_type)
        count_query = count_query.where(Document.source_type == source_type)
    
    if status_filter:
        query = query.where(Document.status == status_filter)
        count_query = count_query.where(Document.status == status_filter)
    
    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination and ordering
    offset = (page - 1) * page_size
    query = query.order_by(Document.created_at.desc()).offset(offset).limit(page_size)
    
    result = await db.execute(query)
    documents = result.scalars().all()
    
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(doc) for doc in documents],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    summary="Get document details",
    description="Returns full metadata for a specific document, including processing status.",
)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a single document's details by its UUID."""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found."
        )
    
    return DocumentResponse.model_validate(doc)


# ══════════════════════════════════════════════════════════════
# DELETE — Remove documents from the knowledge base
# ══════════════════════════════════════════════════════════════


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a document",
    description="Removes a document and all its chunks from both PostgreSQL and Qdrant.",
)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a document completely:
    1. Remove all vector chunks from Qdrant
    2. Remove the metadata row from PostgreSQL
    """
    # Find the document
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found."
        )
    
    # Delete chunks from Qdrant
    qdrant_service.delete_by_document(str(document_id))
    
    # Delete metadata from PostgreSQL
    await db.delete(doc)
    await db.commit()
    
    return {
        "message": f"Document '{doc.filename}' and all its chunks have been deleted.",
        "document_id": str(document_id),
    }


# ══════════════════════════════════════════════════════════════
# CHUNKS — Preview how documents were split
# ══════════════════════════════════════════════════════════════


@router.get(
    "/documents/{document_id}/chunks",
    response_model=ChunkListResponse,
    summary="Preview document chunks",
    description="View the text chunks generated from a document. Useful for verifying ingestion quality.",
)
async def get_document_chunks(
    document_id: uuid.UUID,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """
    Preview chunks for a document — shows how the text was split.
    
    This is a debugging/quality tool. If the RAG gives bad answers,
    check here first: "Were the chunks split sensibly?"
    """
    # Verify the document exists
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found."
        )
    
    # Get chunks from Qdrant
    offset = (page - 1) * page_size
    chunks_data = qdrant_service.get_chunks_by_document(
        document_id=str(document_id),
        limit=page_size,
        offset=offset,
    )
    
    chunks = [
        ChunkPreview(
            chunk_index=c["chunk_index"],
            text=c["text"],
            char_count=c["char_count"],
        )
        for c in chunks_data
    ]
    
    return ChunkListResponse(
        document_id=document_id,
        filename=doc.filename,
        chunks=chunks,
        total_chunks=doc.chunk_count,
        page=page,
        page_size=page_size,
    )


# ══════════════════════════════════════════════════════════════
# SEARCH — Semantic search across all knowledge
# ══════════════════════════════════════════════════════════════


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search the knowledge base",
    description=(
        "Perform a semantic search across all ingested documents. "
        "The query is embedded and compared against all stored chunks "
        "to find the most relevant medical content."
    ),
)
async def search_knowledge(request: SearchRequest):
    """
    Semantic search — returns exactly what /chat would use as its context.

    ⚠️  THIS ENDPOINT USED TO BYPASS THE RAG PIPELINE, AND IT COST US.
    It called qdrant_service.search() directly: no cross-encoder reranking, no
    per-document capping, no adjacent-chunk merging. /chat meanwhile went
    through rag_service.retrieve_context(), which does all three.

    Two divergent retrieval paths meant the evaluation harness — which measures
    this endpoint — reported *byte-identical* results across three runs while
    reranking was demonstrably working on /chat. Every conclusion drawn from
    those numbers was about code no user ever hits.

    Now there is ONE retrieval path. What you measure here is what the model
    receives. An evaluation harness pointed at a different code path than
    production is worse than no harness, because it produces confident numbers
    about the wrong thing.
    """
    chunks = await rag_service.retrieve_context(
        query=request.query,
        limit=request.limit,
        source_type=request.source_type,
    )

    return SearchResponse(
        query=request.query,
        results=[
            SearchResult(
                text=c.text,
                # Cosine similarity, NOT the rerank score. Cross-encoder
                # outputs are unbounded logits and would be meaningless as a
                # "% match". Reranking changes the order, not this number.
                score=c.score,
                document_id=c.document_id,
                filename=c.document_title,
                source_type=c.source_type,
                chunk_index=c.chunk_index,
            )
            for c in chunks
        ],
        total_results=len(chunks),
    )


# ══════════════════════════════════════════════════════════════
# SEED — Populate the knowledge base with curated content
# ══════════════════════════════════════════════════════════════


@router.post(
    "/seed",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Seed the knowledge base with curated radiology content",
    description=(
        "Ingests the built-in curated radiology knowledge (14 articles covering "
        "chest X-ray interpretation, pneumonia, pneumothorax, PE, stroke, trauma, "
        "contrast safety, and structured reporting).\n\n"
        "If `NCBI_EMAIL` is configured in the environment, it also fetches "
        "StatPearls abstracts from the NCBI E-utilities API.\n\n"
        "**Idempotent** — articles already present are skipped, so running this "
        "twice will not create duplicates.\n\n"
        "Runs in the background; returns immediately. Watch the server logs or "
        "poll `GET /knowledge/stats` to see documents appear."
    ),
)
async def seed_knowledge(background_tasks: BackgroundTasks):
    """
    Trigger knowledge base seeding.

    WHY MANUAL RATHER THAN ON STARTUP?
    Seeding embeds ~14 articles, which takes 10-30 seconds. Doing that on every
    container start makes restarts slow and unpredictable, and it fights with
    Docker healthchecks. An explicit endpoint means seeding happens when you
    decide it should.
    """
    _require_embedding_model()

    async def _run_seed() -> None:
        # Fresh session — the request-scoped one is closed by now.
        async with async_session() as db:
            try:
                await seed_knowledge_base(db)
            except Exception as e:  # noqa: BLE001
                await db.rollback()
                print(f"❌ Knowledge base seeding failed: {e}")

    background_tasks.add_task(_run_seed)

    return {
        "message": "Knowledge base seeding started in the background.",
        "curated_articles": len(SEED_KNOWLEDGE),
        "ncbi_fetch_enabled": bool(settings.NCBI_EMAIL),
        "next_step": "Poll GET /api/v1/knowledge/stats to watch documents appear.",
    }


@router.post(
    "/fetch-pmc",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest peer-reviewed articles from PubMed Central Open Access",
    description=(
        "Searches the **PMC Open Access Subset** for radiology topics and "
        "ingests the full text of each article.\n\n"
        "Unlike the curated seed content, every article ingested here carries "
        "a real PMCID, PMID and DOI, and a `source_url` a clinician can open "
        "and verify — which is what makes the evidence panel genuinely "
        "traceable.\n\n"
        "Only articles inside the Open Access Subset are retrieved; anything "
        "whose licence can't be positively confirmed is skipped.\n\n"
        "**Idempotent** — articles already ingested are skipped, so running "
        "this repeatedly tops up the corpus rather than duplicating it.\n\n"
        "Runs in the background. Expect roughly 1–3 minutes for the default "
        "topic list; poll `GET /knowledge/stats` to watch it grow."
    ),
)
async def fetch_pmc(
    background_tasks: BackgroundTasks,
    request: PMCFetchRequest | None = None,
):
    """
    Trigger PMC Open Access ingestion.

    Body (all optional):
        {"topics": ["pneumothorax chest imaging", ...], "max_per_topic": 15}

    Omit the body entirely to use the default radiology topic list.
    """
    _require_embedding_model()

    if not ncbi_is_configured():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "NCBI_EMAIL is not set to a real address. NCBI's terms of use "
                "require a contactable email on every request. Set NCBI_EMAIL "
                "in backend/.env and restart the backend."
            ),
        )

    req = request or PMCFetchRequest()
    max_per_topic = req.max_per_topic
    topic_list = [t.strip() for t in (req.topics or []) if t.strip()] or None

    async def _run() -> None:
        async with async_session() as db:
            try:
                summary = await fetch_and_ingest_pmc(
                    db, topics=topic_list, max_per_topic=max_per_topic
                )
                print(f"📚 PMC ingestion complete: {summary}")
            except Exception as e:  # noqa: BLE001
                await db.rollback()
                print(f"❌ PMC ingestion failed: {e}")

    background_tasks.add_task(_run)

    return {
        "message": "PMC Open Access ingestion started in the background.",
        "topics": topic_list or DEFAULT_TOPICS,
        "max_per_topic": max_per_topic,
        "next_step": "Poll GET /api/v1/knowledge/stats to watch documents appear.",
    }


# ══════════════════════════════════════════════════════════════
# STATS — Knowledge base overview
# ══════════════════════════════════════════════════════════════


@router.get(
    "/stats",
    response_model=KnowledgeBaseStats,
    summary="Knowledge base statistics",
    description="Returns an overview of the knowledge base: document counts, chunk counts, and source type breakdown.",
)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """
    Dashboard statistics — gives a quick overview of the knowledge base health.
    
    Shows:
    - Total documents and their status breakdown
    - Total chunks across all documents
    - Source type distribution (how many textbooks, guidelines, etc.)
    - Qdrant vector count (should match total chunks)
    """
    # Count documents by status
    total_result = await db.execute(select(func.count(Document.id)))
    total = total_result.scalar() or 0
    
    completed_result = await db.execute(
        select(func.count(Document.id)).where(Document.status == "completed")
    )
    completed = completed_result.scalar() or 0
    
    failed_result = await db.execute(
        select(func.count(Document.id)).where(Document.status == "failed")
    )
    failed = failed_result.scalar() or 0
    
    processing_result = await db.execute(
        select(func.count(Document.id)).where(Document.status == "processing")
    )
    processing = processing_result.scalar() or 0
    
    # Total chunks across all completed documents
    chunks_result = await db.execute(
        select(func.sum(Document.chunk_count)).where(Document.status == "completed")
    )
    total_chunks = chunks_result.scalar() or 0
    
    # Source type breakdown
    source_counts_result = await db.execute(
        select(Document.source_type, func.count(Document.id))
        .group_by(Document.source_type)
    )
    source_type_counts = {row[0]: row[1] for row in source_counts_result.all()}
    
    # Get Qdrant vector count
    qdrant_info = qdrant_service.get_collection_info()
    qdrant_vectors = qdrant_info.get("vectors_count", 0)
    
    return KnowledgeBaseStats(
        total_documents=total,
        completed_documents=completed,
        failed_documents=failed,
        processing_documents=processing,
        total_chunks=total_chunks,
        source_type_counts=source_type_counts,
        qdrant_vectors=qdrant_vectors,
    )
