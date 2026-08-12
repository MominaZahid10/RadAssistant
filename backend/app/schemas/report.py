"""
RadAssist AI — Report Schemas (Phase 5, Step 0.5)

Request and response shapes for the report sign-off loop.
"""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ReportStatusEnum(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReportCreate(BaseModel):
    """Persist a draft the model just produced."""

    findings_input: str = Field(
        min_length=1, max_length=20_000,
        description="The findings the clinician dictated, verbatim.",
    )
    ai_draft: str = Field(
        min_length=1, max_length=50_000,
        description=(
            "The generated draft, exactly as produced. Stored immutably — "
            "edits go to edited_text so the model's output and the human's "
            "corrections stay distinguishable."
        ),
    )
    model: str | None = Field(
        default=None,
        description="Which model produced the draft, for later explanation.",
    )
    sources: list[dict] | None = Field(
        default=None,
        description=(
            "The retrieved chunks the draft was generated against. Kept "
            "because the corpus gets re-indexed and the model gets swapped; "
            "without a snapshot, 'what informed this report' is unanswerable."
        ),
    )
    image_id: UUID | None = None


class ReportUpdate(BaseModel):
    """
    Edit the text, or sign the report off.

    ⚠️  STATUS AND TEXT MOVE TOGETHER ON PURPOSE.
    Approving is a claim about a specific wording. If the editor sent the text
    in one request and the approval in another, a slow network could record an
    approval against text the reviewer never saw. One request, one decision.
    """

    edited_text: str | None = Field(
        default=None, max_length=50_000,
        description=(
            "The reviewer's wording. Omit to accept the draft unchanged — "
            "that is a distinct, meaningful state, not the same as sending "
            "back an identical string."
        ),
    )
    status: ReportStatusEnum | None = None
    reviewed_by: str | None = Field(
        default=None, max_length=200,
        description="Who signed it. Free text until Phase 6 adds auth.",
    )
    review_note: str | None = Field(
        default=None, max_length=5_000,
        description="Why it was rejected, or a note attached at sign-off.",
    )


class ReportResponse(BaseModel):
    id: UUID

    findings_input: str
    ai_draft: str = Field(description="The model's original output. Never modified.")
    edited_text: str | None = None
    final_text: str = Field(
        description="What the report says: edits if present, otherwise the draft."
    )
    was_edited: bool = Field(
        description=(
            "Whether a human changed the wording. Compared on content, so "
            "opening a draft and approving it unchanged does not count as an "
            "edit."
        )
    )

    status: str
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_note: str | None = None

    model: str | None = None
    sources: list[dict] | None = None
    image_id: UUID | None = None

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class ReportListResponse(BaseModel):
    reports: list[ReportResponse]
    total: int
    page: int
    page_size: int


class ReportStats(BaseModel):
    """
    Sign-off statistics.

    `edit_rate` is the number that matters. It is how often the model's draft
    needed changing before a clinician would sign it — the closest thing this
    project has to a measured safety signal, and it only exists because
    ai_draft and edited_text are stored separately.
    """
    total: int
    draft: int
    approved: int
    rejected: int
    edited: int
    edit_rate: float = Field(
        description="Fraction of reviewed reports a human had to reword."
    )
    approval_rate: float = Field(
        description="Fraction of reviewed reports that were approved."
    )
