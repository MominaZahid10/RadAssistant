"""
RadAssist AI — Medical Image Model (Phase 4)

Tracks every image in the system: DICOM studies, photographs of reports
uploaded by clinicians, and figures extracted from PMC Open Access articles.

⚠️  FILES ON DISK, METADATA IN POSTGRES — NEVER IMAGE BLOBS IN THE DATABASE.
A single chest CT series is 100-500 MB. Storing that in Postgres bloats every
backup, breaks streaming replication, and makes unrelated queries slower
because the row data no longer fits in cache. Standard practice is a
filesystem (later: object storage) with the path recorded here.

THREE KINDS OF IMAGE, ONE TABLE:

    dicom_upload   a real study — carries PHI, must be de-identified
    report_upload  a photo or scan of a paper report — OCR'd to text
    pmc_figure     a figure from an open-access article, with its caption

They share enough structure that separate tables would mean three near-
identical schemas and three sets of queries. `source_type` distinguishes them,
matching how `documents` already works.

RELATIONSHIP TO documents:
    pmc_figure     → document_id set: the article the figure came from
    report_upload  → document_id set: the text extracted from it by OCR
    dicom_upload   → document_id NULL: a study isn't a text document

So `document_id` is nullable by design, not by omission.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    DateTime,
    Date,
    Text,
    ForeignKey,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database import Base


class MedicalImage(Base):
    """
    One image, wherever it came from.

    LIFECYCLE:
    1. Uploaded or extracted   → status="processing"
    2. Parsed, de-identified,
       thumbnail generated     → status="completed"
    3. Anything fails          → status="failed", error_message explains
    """

    __tablename__ = "medical_images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # ── Link to a text document, where one exists ────────────
    # ondelete="SET NULL": deleting an article must not silently delete its
    # figures. Orphaned images are recoverable; destroyed ones are not.
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ── File ─────────────────────────────────────────────────
    filename = Column(String(500), nullable=False)
    # Relative to IMAGE_DIR — never absolute. An absolute path breaks the
    # moment the storage root moves, and it will move (container → volume →
    # object storage).
    storage_path = Column(String(1000), nullable=False)
    thumbnail_path = Column(String(1000), nullable=True)
    mime_type = Column(String(100), nullable=False, default="image/png")
    file_size = Column(Integer, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)

    # ── Clinical metadata ────────────────────────────────────
    # From DICOM tags where available, otherwise user-supplied or inferred.
    # All nullable: a PMC figure has none of this.
    modality = Column(String(20), nullable=True, index=True)      # CR, DX, CT, MR, US
    body_part = Column(String(100), nullable=True)                 # CHEST, ABDOMEN
    view_position = Column(String(20), nullable=True)              # PA, AP, LATERAL
    # Date only, never datetime — DICOM StudyDate has no time component, and
    # storing a fake midnight implies precision that isn't there.
    study_date = Column(Date, nullable=True)

    # ── Provenance ───────────────────────────────────────────
    # 'dicom_upload' | 'report_upload' | 'pmc_figure' | 'image_upload'
    source_type = Column(String(50), nullable=False, default="image_upload", index=True)
    source_url = Column(String(2000), nullable=True)
    # For PMC figures this is the caption — the text half of an image-text
    # pair, and what makes later multimodal retrieval possible.
    caption = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    # ⚠️  REDISTRIBUTION TERMS, RECORDED PER IMAGE — NOT ASSUMED.
    # We only ingest from the PMC Open Access Subset, so reuse is permitted.
    # But "open access" is not one licence: CC-BY permits derivatives with
    # attribution, CC-BY-NC-ND permits neither commercial use nor derivatives.
    # Checking at ingest time and not recording the answer leaves a claim
    # nobody can verify later. Nullable — uploads have no upstream licence.
    licence = Column(String(100), nullable=True)

    # ── De-identification ────────────────────────────────────
    # ⚠️  DEFAULTS TO FALSE ON PURPOSE.
    # An image is only marked de-identified after the process has demonstrably
    # run. Defaulting to True would mean a bug in the pipeline silently
    # produces images labelled safe that still carry PHI — the worst possible
    # failure in this phase.
    is_deidentified = Column(Boolean, nullable=False, default=False)
    # Retained DICOM tags — allowlisted only, never the raw tag set.
    dicom_metadata = Column(JSONB, nullable=True)

    # ── Extracted text (report photos) ───────────────────────
    ocr_text = Column(Text, nullable=True)

    # ── Processing status ────────────────────────────────────
    status = Column(String(20), nullable=False, default="processing", index=True)
    error_message = Column(Text, nullable=True)

    # ── Timestamps ───────────────────────────────────────────
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

    # Composite index for the gallery's common filter: "chest X-rays" is
    # modality + body_part together, and two single-column indexes can't
    # serve that as efficiently as one composite.
    __table_args__ = (
        Index("ix_medical_images_modality_body_part", "modality", "body_part"),
    )

    def __repr__(self) -> str:
        return f"<MedicalImage {self.filename} [{self.source_type}/{self.status}]>"
