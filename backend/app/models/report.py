"""
RadAssist AI — Report Model (Phase 5, Step 0.5)

A drafted radiology report and its review trail.

════════════════════════════════════════════════════════════════════
WHY THIS TABLE EXISTS AT ALL
════════════════════════════════════════════════════════════════════
The project document specifies a Review & Sign-off Layer — *"the radiologist
edits, approves, or rejects"* — and an explicit design constraint: the system
produces drafts only, and the UI must make the human-in-the-loop visible.

Before this, a draft appeared as a chat message. It could not be edited,
approved, or retrieved. The "human in the loop" was a sentence printed at the
bottom of the output, which is a claim rather than a mechanism.

════════════════════════════════════════════════════════════════════
THE AI DRAFT AND THE HUMAN TEXT ARE SEPARATE COLUMNS, PERMANENTLY
════════════════════════════════════════════════════════════════════
`ai_draft` is written once and never modified. `edited_text` holds whatever the
radiologist actually signed. Overwriting the first with the second would be
simpler and would destroy the only record of what the model produced versus
what a human corrected.

That difference is the evidence for two things this project claims:
  - the safety argument — how often, and how badly, does the model need
    correcting before a clinician will sign it?
  - the efficiency metric in the project document — time saved per report only
    means something if you know how much of the draft survived.

A single mutable `text` column answers neither, and the information cannot be
reconstructed after the fact.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.core.database import Base


class ReportStatus:
    """
    Lifecycle of a draft.

    Deliberately not a database enum: adding a state to a Postgres enum needs
    a migration and locks the table, and this lifecycle will grow (amended,
    superseded, addendum). A constrained string costs one CHECK and stays
    cheap to extend.
    """
    DRAFT = "draft"          # generated, not yet reviewed
    APPROVED = "approved"    # a human signed it
    REJECTED = "rejected"    # a human refused it — kept, not deleted

    ALL = (DRAFT, APPROVED, REJECTED)


class Report(Base):
    """A generated report draft, its edits, and its sign-off."""

    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # ── What the clinician dictated ──────────────────────────
    # The input, kept verbatim. Regeneration needs it, and it is the only way
    # to tell later whether a bad draft came from a bad model or thin input.
    findings_input = Column(Text, nullable=False)

    # ── What the model produced ──────────────────────────────
    # ⚠️  IMMUTABLE. Never updated after creation — see the module docstring.
    ai_draft = Column(Text, nullable=False)

    # ── What the human signed ────────────────────────────────
    # NULL means untouched: the radiologist accepted the draft as written.
    # That is a meaningful value, distinct from "edited to be identical".
    edited_text = Column(Text, nullable=True)

    status = Column(String(20), nullable=False, default=ReportStatus.DRAFT, index=True)

    # ── Review trail ─────────────────────────────────────────
    # No auth yet (Phase 6), so reviewer is free text for now. The column
    # exists so sign-off has somewhere to land the moment auth arrives —
    # retrofitting an audit trail after the fact means the early records
    # simply have no reviewer, forever.
    reviewed_by = Column(String(200), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    review_note = Column(Text, nullable=True)

    # ── Provenance of the draft ──────────────────────────────
    # Which model wrote it, and which chunks it saw. Kept so a report can be
    # explained months later, when the model has been swapped and the corpus
    # re-indexed. Without this, "which sources informed this report" becomes
    # unanswerable — and that traceability is the product.
    model = Column(String(200), nullable=True)
    sources = Column(JSONB, nullable=True)

    # Optional link to an uploaded study or report photo.
    image_id = Column(
        UUID(as_uuid=True),
        ForeignKey("medical_images.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    @property
    def final_text(self) -> str:
        """What the report actually says — edits if any, otherwise the draft."""
        return self.edited_text if self.edited_text is not None else self.ai_draft

    @property
    def was_edited(self) -> bool:
        """
        True if a human changed the wording.

        Compared on content, not on the column being set: the editor writes
        back on every save, so `edited_text is not None` would report an edit
        for a report that was opened and approved unchanged.
        """
        return (
            self.edited_text is not None
            and self.edited_text.strip() != (self.ai_draft or "").strip()
        )

    def __repr__(self) -> str:
        return f"<Report {self.id} status={self.status} edited={self.was_edited}>"
