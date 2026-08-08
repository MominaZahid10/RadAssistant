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
    POST   /api/v1/knowledge/upload              Upload a file
    GET    /api/v1/knowledge/documents            List all documents
    GET    /api/v1/knowledge/documents/{id}       Get one document's details
    DELETE /api/v1/knowledge/documents/{id}       Delete a document
    GET    /api/v1/knowledge/documents/{id}/chunks Preview chunks
    POST   /api/v1/knowledge/search               Semantic search
    GET    /api/v1/knowledge/stats                 Knowledge base statistics
"""

import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.database import get_db
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
from app.services.ingestion import ingest_document, ingest_text_content
from app.services.embedding import embedding_service
from app.services.qdrant_service import qdrant_service

settings = get_settings()

# Create the router — all endpoints here get the /knowledge prefix
router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])


# ══════════════════════════════════════════════════════════════
# UPLOAD — Accept files from doctors
# ══════════════════════════════════════════════════════════════


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document to the knowledge base",
    description=(
        "Upload a PDF, DOCX, TXT, or image file. The file will be parsed, "
        "chunked, embedded, and stored in the vector database for semantic search. "
        "Supported formats: PDF, DOCX, TXT, MD, PNG, JPG, JPEG, TIFF, BMP."
    ),
)
async def upload_document(
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
    Upload a document and process it through the ingestion pipeline.
    
    FLOW:
    1. Validate file type and size
    2. Create a document record in PostgreSQL (status: "processing")
    3. Run the ingestion pipeline (parse → chunk → embed → store)
    4. Update the document status to "completed" or "failed"
    5. Return the document ID and status
    
    The frontend can then use the document ID to check status,
    view chunks, or delete the document.
    """
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
    
    # ── Run the ingestion pipeline ──────────────────────────
    result = await ingest_document(
        file_bytes=file_bytes,
        filename=file.filename,
        file_type=file_ext.lstrip("."),
        document_id=str(doc.id),
        source_type=source_type,
        title=title,
    )
    
    # ── Update document status in PostgreSQL ────────────────
    doc.status = result["status"]
    doc.chunk_count = result["chunk_count"]
    if result["status"] == "failed":
        doc.error_message = result["message"]
    doc.updated_at = datetime.now(timezone.utc)
    
    await db.commit()
    await db.refresh(doc)
    
    return DocumentUploadResponse(
        id=doc.id,
        filename=doc.filename,
        status=doc.status,
        message=result["message"],
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
    Semantic search — find relevant medical content by meaning.
    
    THIS IS THE PREVIEW OF PHASE 3's RAG RETRIEVAL.
    
    How it works:
    1. Your query text is converted to a 384-dim vector
    2. Qdrant finds the stored chunks with the most similar vectors
    3. Results are returned with similarity scores and source info
    
    Example query: "What are the radiographic findings of pneumothorax?"
    → Returns chunks about absent lung markings, visible pleural line, etc.
    """
    # Embed the search query
    query_vector = embedding_service.encode_single(request.query)
    
    # Search Qdrant
    results = qdrant_service.search(
        query_vector=query_vector,
        limit=request.limit,
        source_type=request.source_type,
    )
    
    return SearchResponse(
        query=request.query,
        results=[
            SearchResult(
                text=r["text"],
                score=r["score"],
                document_id=r.get("document_id"),
                filename=r.get("filename"),
                source_type=r.get("source_type"),
                chunk_index=r.get("chunk_index"),
            )
            for r in results
        ],
        total_results=len(results),
    )


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
