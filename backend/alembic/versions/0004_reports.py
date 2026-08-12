"""Add reports table

Revision ID: 0004_reports
Revises: 0003_image_licence
Created: 2026-08-12

The Review & Sign-off Layer from the project document. Drafts were previously
chat messages — unstoreable, uneditable, unapprovable — so "the radiologist
edits, approves, or rejects" had no mechanism behind it.

⚠️  ai_draft AND edited_text ARE SEPARATE COLUMNS AND MUST STAY THAT WAY.
Collapsing them into one mutable `text` field would be simpler and would
destroy the only record of what the model wrote versus what a human corrected
before signing. That delta is the evidence for the project's safety claim and
for its efficiency metric, and it cannot be reconstructed later.

IDEMPOTENT, like 0001 and 0003 — these run on every container start.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004_reports"
down_revision = "0003_image_licence"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if _has_table("reports"):
        return

    op.create_table(
        "reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),

        # The dictation, verbatim. Needed to regenerate, and to tell a bad
        # model apart from thin input when reviewing a poor draft.
        sa.Column("findings_input", sa.Text(), nullable=False),

        # ⚠️  Written once, never updated.
        sa.Column("ai_draft", sa.Text(), nullable=False),

        # NULL means the draft was accepted as written — a distinct and
        # meaningful state, not the same as "edited to be identical".
        sa.Column("edited_text", sa.Text(), nullable=True),

        sa.Column("status", sa.String(length=20), nullable=False,
                  server_default="draft"),

        # Review trail. Free text until Phase 6 brings auth — the columns
        # exist now so sign-off has somewhere to land, because retrofitting an
        # audit trail leaves every early record without a reviewer forever.
        sa.Column("reviewed_by", sa.String(length=200), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("review_note", sa.Text(), nullable=True),

        # Which model wrote it, and what it saw. Without this, "which sources
        # informed this report" is unanswerable once the corpus is re-indexed.
        sa.Column("model", sa.String(length=200), nullable=True),
        sa.Column("sources", postgresql.JSONB(astext_type=sa.Text()), nullable=True),

        # SET NULL, not CASCADE: deleting an image must not destroy the report
        # that was written about it.
        sa.Column("image_id", postgresql.UUID(as_uuid=True), nullable=True),

        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),

        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["image_id"], ["medical_images.id"],
                                ondelete="SET NULL"),

        # A status outside the lifecycle is a bug, and a bug that reaches the
        # database outlives the deploy that caused it. Enforced here as well
        # as in the API schema.
        sa.CheckConstraint(
            "status IN ('draft', 'approved', 'rejected')",
            name="ck_reports_status",
        ),
    )

    op.create_index("ix_reports_id", "reports", ["id"])
    op.create_index("ix_reports_status", "reports", ["status"])
    op.create_index("ix_reports_image_id", "reports", ["image_id"])
    # Listing is always newest-first; without this it is a sort over the table.
    op.create_index("ix_reports_created_at", "reports", ["created_at"])


def downgrade() -> None:
    if _has_table("reports"):
        op.drop_table("reports")
