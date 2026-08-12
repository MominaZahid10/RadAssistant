"""
RadAssist AI — Image Schemas (Phase 4)

Request and response shapes for the image API.

⚠️  NOTE WHAT IS NOT EXPOSED.
`storage_path` and `thumbnail_path` are internal. Returning them would leak
the server's filesystem layout and invite clients to construct their own URLs
— which is exactly how a path-traversal endpoint gets built by accident.
Clients get `/images/{id}/file` and let the server resolve it.
"""

from datetime import date, datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ImageSourceType(str, Enum):
    """
    Where an image came from. Determines how it was processed and what
    metadata it carries.
    """
    DICOM_UPLOAD = "dicom_upload"      # a study — de-identified on ingest
    REPORT_UPLOAD = "report_upload"    # photo/scan of a paper report — OCR'd
    PMC_FIGURE = "pmc_figure"          # figure from an open-access article
    IMAGE_UPLOAD = "image_upload"      # anything else


# ══════════════════════════════════════════════════════════════
# RESPONSES
# ══════════════════════════════════════════════════════════════


class ImageResponse(BaseModel):
    """A single image's metadata."""

    id: UUID
    document_id: UUID | None = Field(
        default=None,
        description=(
            "The text document this image relates to, if any. PMC figures "
            "link to their article; OCR'd reports link to their extracted "
            "text; DICOM studies link to nothing."
        ),
    )

    filename: str
    mime_type: str
    file_size: int | None = None
    width: int | None = None
    height: int | None = None

    # ── Clinical metadata (null for figures) ──
    modality: str | None = Field(default=None, description="CR, DX, CT, MR, US")
    body_part: str | None = None
    view_position: str | None = Field(default=None, description="PA, AP, LATERAL")
    study_date: date | None = Field(
        default=None,
        description=(
            "Reduced to 1 January of the study year. Dates finer than a year "
            "are quasi-identifiers under HIPAA Safe Harbor."
        ),
    )

    # ── Provenance ──
    source_type: str
    source_url: str | None = None
    caption: str | None = Field(
        default=None,
        description="Figure caption — the text half of an image-text pair.",
    )
    description: str | None = None
    licence: str | None = Field(
        default=None,
        description=(
            "Redistribution terms, e.g. 'CC-BY 4.0'. Recorded per image "
            "rather than assumed from the source: 'open access' covers "
            "licences that do and do not permit derivatives. Null for "
            "user uploads, which have no upstream licence."
        ),
    )

    is_deidentified: bool = Field(
        description=(
            "True only after de-identification has demonstrably run. Never "
            "assumed."
        ),
    )
    dicom_metadata: dict | None = Field(
        default=None,
        description="Allowlisted DICOM tags only — never the raw tag set.",
    )
    ocr_text: str | None = None

    status: str = Field(description="processing | completed | failed")
    error_message: str | None = None

    created_at: datetime
    updated_at: datetime

    # URLs the client should use. The server resolves these to disk paths;
    # clients never see or construct filesystem paths.
    file_url: str | None = None
    thumbnail_url: str | None = None

    model_config = ConfigDict(from_attributes=True)


class ImageListResponse(BaseModel):
    images: list[ImageResponse]
    total: int
    page: int
    page_size: int


class ImageUploadResponse(BaseModel):
    """
    Returned immediately on upload, before processing finishes.

    Same pattern as document upload: parsing, de-identification and thumbnail
    generation can take seconds for a large study, so the request returns at
    once and the client polls.
    """
    id: UUID
    filename: str
    status: str
    message: str
    detected_type: str = Field(
        description="What the file was detected as: dicom, image, or report."
    )


class ImageStats(BaseModel):
    """Overview for the dashboard."""
    total_images: int
    completed: int
    failed: int
    processing: int
    by_source_type: dict[str, int]
    by_modality: dict[str, int]
    deidentified_count: int
    storage_bytes: int
    storage_files: int
