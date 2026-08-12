"""Add medical_images table

Revision ID: 0002_medical_images
Revises: 0001_initial
Created: 2026-08-11

THIS IS THE MIGRATION create_all() WOULD HAVE HANDLED WRONG.

create_all() creates missing tables, so this one it would actually have made.
But the moment Phase 4 adds a column to `documents` — or changes a type, or
adds a constraint — create_all does nothing at all, silently, and the app then
fails at query time with an error that points nowhere near the cause. Step 0
existed so this and every change after it are explicit and reversible.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_medical_images"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "medical_images",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),

        # Nullable BY DESIGN: a DICOM study isn't a text document. PMC figures
        # and OCR'd report photos link to one; studies don't.
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),

        # ── File ──
        sa.Column("filename", sa.String(length=500), nullable=False),
        # Relative to IMAGE_DIR. Absolute paths break when storage moves.
        sa.Column("storage_path", sa.String(length=1000), nullable=False),
        sa.Column("thumbnail_path", sa.String(length=1000), nullable=True),
        sa.Column("mime_type", sa.String(length=100), nullable=False,
                  server_default="image/png"),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),

        # ── Clinical metadata (all nullable — a PMC figure has none) ──
        sa.Column("modality", sa.String(length=20), nullable=True),
        sa.Column("body_part", sa.String(length=100), nullable=True),
        sa.Column("view_position", sa.String(length=20), nullable=True),
        # Date, not DateTime: DICOM StudyDate has no time component and a
        # fake midnight implies precision that doesn't exist.
        sa.Column("study_date", sa.Date(), nullable=True),

        # ── Provenance ──
        sa.Column("source_type", sa.String(length=50), nullable=False,
                  server_default="image_upload"),
        sa.Column("source_url", sa.String(length=2000), nullable=True),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),

        # ── De-identification ──
        # server_default false: an image is only marked safe once the process
        # has demonstrably run. Defaulting true would mean a pipeline bug
        # silently labels PHI-carrying images as de-identified.
        sa.Column("is_deidentified", sa.Boolean(), nullable=False,
                  server_default=sa.false()),
        sa.Column("dicom_metadata", postgresql.JSONB(), nullable=True),

        # ── Extracted text ──
        sa.Column("ocr_text", sa.Text(), nullable=True),

        # ── Status ──
        sa.Column("status", sa.String(length=20), nullable=False,
                  server_default="processing"),
        sa.Column("error_message", sa.Text(), nullable=True),

        # ── Timestamps ──
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),

        sa.PrimaryKeyConstraint("id"),
        # SET NULL, not CASCADE: deleting an article must not silently destroy
        # its figures. An orphaned image is recoverable; a deleted one isn't.
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"],
            ondelete="SET NULL",
            name="fk_medical_images_document_id",
        ),
    )

    op.create_index(op.f("ix_medical_images_id"), "medical_images", ["id"])
    op.create_index(op.f("ix_medical_images_document_id"), "medical_images", ["document_id"])
    op.create_index(op.f("ix_medical_images_modality"), "medical_images", ["modality"])
    op.create_index(op.f("ix_medical_images_source_type"), "medical_images", ["source_type"])
    op.create_index(op.f("ix_medical_images_status"), "medical_images", ["status"])
    # The gallery filters on modality AND body_part together ("chest X-rays"),
    # which two single-column indexes serve less well than one composite.
    op.create_index(
        "ix_medical_images_modality_body_part",
        "medical_images",
        ["modality", "body_part"],
    )


def downgrade() -> None:
    op.drop_index("ix_medical_images_modality_body_part", table_name="medical_images")
    op.drop_index(op.f("ix_medical_images_status"), table_name="medical_images")
    op.drop_index(op.f("ix_medical_images_source_type"), table_name="medical_images")
    op.drop_index(op.f("ix_medical_images_modality"), table_name="medical_images")
    op.drop_index(op.f("ix_medical_images_document_id"), table_name="medical_images")
    op.drop_index(op.f("ix_medical_images_id"), table_name="medical_images")
    op.drop_table("medical_images")
