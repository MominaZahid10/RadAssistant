"""
RadAssist AI — Document Model (PostgreSQL)

WHAT THIS FILE DOES:
Defines the 'documents' table in PostgreSQL. Every time a doctor uploads
a file or we ingest a medical article, a row is created here to track it.

WHY POSTGRESQL AND NOT JUST QDRANT?
Qdrant is great at SEARCHING vectors, but terrible at:
- Listing all documents with pagination
- Filtering by upload date, status, file type
- Tracking processing status (pending → processing → completed → failed)
- Auditing who uploaded what and when

So we use BOTH:
- PostgreSQL = metadata (the "card catalog" of our library)
- Qdrant = actual text chunks + vectors (the "books" themselves)

WHAT IS A MODEL?
A Python class that maps directly to a database table.
Each attribute becomes a column. SQLAlchemy handles the SQL for us.
    Python class "Document"  →  SQL table "documents"
    Python attribute "filename"  →  SQL column "filename"

WHAT IS UUID?
A Universally Unique Identifier — a random 128-bit ID like:
    "550e8400-e29b-41d4-a716-446655440000"
Unlike auto-incrementing integers (1, 2, 3...), UUIDs:
- Can be generated anywhere without a central counter
- Don't reveal how many records exist (security)
- Are standard in medical/enterprise systems
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class Document(Base):
    """
    Tracks every document in the knowledge base.
    
    LIFECYCLE:
    1. Doctor uploads a PDF  →  row created with status="pending"
    2. Ingestion starts      →  status changes to "processing"
    3. Text extracted, chunked, embedded  →  status="completed"
    4. If anything fails     →  status="failed", error_message explains why
    
    The actual text chunks and their vectors are stored in Qdrant,
    linked back to this record via the document's UUID.
    """

    __tablename__ = "documents"

    # ── Primary Key ──────────────────────────────────────────
    # UUID is generated in Python (not by the database) so we
    # have the ID immediately without waiting for a DB round-trip.
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )

    # ── File Information ─────────────────────────────────────
    # What was uploaded and how big is it?
    filename = Column(String(500), nullable=False)       # Original filename
    file_type = Column(String(20), nullable=False)       # "pdf", "docx", "txt", "png", etc.
    file_size = Column(Integer, nullable=True)            # Size in bytes

    # ── Content Metadata ─────────────────────────────────────
    # What kind of medical content is this?
    title = Column(String(1000), nullable=True)           # Document title (extracted or user-provided)
    source_type = Column(
        String(50),
        nullable=False,
        default="general",
        # Valid values: "textbook", "guideline", "report", "research_paper",
        #               "statpearls", "clinical_note", "general"
        # We don't use a DB enum so we can add new types without migrations
    )
    source_url = Column(String(2000), nullable=True)     # If fetched from the internet (e.g., NCBI URL)
    description = Column(Text, nullable=True)             # Optional description/notes

    # ── Processing Status ────────────────────────────────────
    # Tracks where the document is in the ingestion pipeline.
    status = Column(
        String(20),
        nullable=False,
        default="pending",
        index=True,
        # Valid values: "pending", "processing", "completed", "failed"
    )
    chunk_count = Column(Integer, default=0)              # How many chunks were created
    error_message = Column(Text, nullable=True)           # If status="failed", why?

    # ── Timestamps ───────────────────────────────────────────
    # Always know WHEN something happened — critical for auditing.
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self):
        """How this object appears in logs/debugging."""
        return f"<Document {self.filename} [{self.status}]>"
