"""
RadAssist AI — Report Schemas (Phase 3)

Request/response shapes for the report generation endpoint.

REPORT MODE IS DIFFERENT FROM CHAT:
- Chat (Q&A): Explains reasoning, defines terms, conversational
- Report:     Terse clinical register, structured Findings/Impression,
              no explanatory prose — output goes into a medical record

These schemas enforce that distinction at the API boundary.
"""

from pydantic import BaseModel, Field
from enum import Enum


# ══════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════


class Modality(str, Enum):
    """
    Imaging modality — used to give the LLM context about what kind
    of study produced these findings. Affects terminology and structure.
    """
    XRAY = "x-ray"
    CT = "ct"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    MAMMOGRAPHY = "mammography"
    FLUOROSCOPY = "fluoroscopy"
    PET_CT = "pet-ct"
    OTHER = "other"


class BodyRegion(str, Enum):
    """Body region being imaged — helps the LLM scope its knowledge retrieval."""
    CHEST = "chest"
    ABDOMEN = "abdomen"
    HEAD = "head"
    SPINE = "spine"
    MSK = "musculoskeletal"
    BREAST = "breast"
    CARDIAC = "cardiac"
    VASCULAR = "vascular"
    PELVIS = "pelvis"
    NECK = "neck"
    OTHER = "other"


# ══════════════════════════════════════════════════════════════
# REQUEST
# ══════════════════════════════════════════════════════════════


class ReportRequest(BaseModel):
    """
    Request to generate a structured radiology report section.

    Example:
    {
        "findings": "PA and lateral chest radiograph. Heart size normal.
                     Lungs are clear. No pleural effusion or pneumothorax.
                     No acute osseous abnormality.",
        "modality": "x-ray",
        "body_region": "chest",
        "include_sources": true
    }
    """
    findings: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description=(
            "The clinical findings to generate a report from. "
            "Can be free-text observations, key findings, or "
            "a draft that needs structuring."
        ),
    )
    modality: Modality = Field(
        default=Modality.OTHER,
        description="Imaging modality (x-ray, ct, mri, etc.)",
    )
    body_region: BodyRegion = Field(
        default=BodyRegion.OTHER,
        description="Body region being imaged.",
    )
    include_sources: bool = Field(
        default=True,
        description=(
            "If true, the response includes the source chunks "
            "used to ground the generated report."
        ),
    )


# ══════════════════════════════════════════════════════════════
# SOURCE (reuse from rag schemas)
# ══════════════════════════════════════════════════════════════

# We import SourceReference from rag.py — no duplication.
from app.schemas.rag import SourceReference  # noqa: E402


# ══════════════════════════════════════════════════════════════
# RESPONSE
# ══════════════════════════════════════════════════════════════


class ReportResponse(BaseModel):
    """
    Generated radiology report section.

    Example:
    {
        "report": "FINDINGS:\\nHeart size is normal...\\n\\nIMPRESSION:\\n1. No acute...",
        "sources": [...],
        "modality": "x-ray",
        "body_region": "chest",
        "model": "llama-3.3-70b-versatile"
    }
    """
    report: str = Field(
        description="The generated report text (Findings/Impression format).",
    )
    sources: list[SourceReference] | None = Field(
        default=None,
        description="Source chunks used to ground the report.",
    )
    modality: str = Field(
        description="The modality of the study.",
    )
    body_region: str = Field(
        description="The body region imaged.",
    )
    model: str = Field(
        description="The LLM model that generated the report.",
    )
