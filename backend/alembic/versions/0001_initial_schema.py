"""Initial schema — captures the documents table as built by create_all

Revision ID: 0001_initial
Revises:
Created: 2026-08-11

⚠️  THIS MIGRATION IS DELIBERATELY IDEMPOTENT.

Until now the schema was built by `Base.metadata.create_all()` at startup, so
existing databases already contain `documents` while fresh ones don't. A plain
`op.create_table()` would crash on every existing deployment with
"relation already exists".

Rather than requiring a manual `alembic stamp head` — a step that WILL be
forgotten, and whose failure mode is a confusing crash on someone else's
machine — this checks first and skips creation when the table is present. The
result is the same either way: the database matches the model, and
alembic_version records that 0001 has been applied.

WHY MOVE OFF create_all AT ALL:
It creates tables but never ALTERS them. Add a column to an existing model and
create_all does nothing, silently — no error, no change, and the app then
fails at query time with a missing-column error that points nowhere near the
cause. Phase 4 adds `medical_images` and will add columns to `documents`, so
this is the last moment it's cheap to fix.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if _table_exists("documents"):
        # Pre-existing database from the create_all era. Nothing to do —
        # recording this revision as applied is the whole point.
        print("  ℹ️  'documents' already exists — adopting it into Alembic")
        return

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),

        # ── File information ──
        sa.Column("filename", sa.String(length=500), nullable=False),
        sa.Column("file_type", sa.String(length=20), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=True),

        # ── Content metadata ──
        sa.Column("title", sa.String(length=1000), nullable=True),
        # Not a DB enum: source types are added regularly (curated_summary,
        # pmc_open_access, statpearls...) and an enum would need a migration
        # every time.
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("source_url", sa.String(length=2000), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),

        # ── Processing status ──
        # pending → processing → completed | failed
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),

        # ── Timestamps ──
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),

        sa.PrimaryKeyConstraint("id"),
    )

    # Matches the indexes declared on the model.
    op.create_index(op.f("ix_documents_id"), "documents", ["id"])
    op.create_index(op.f("ix_documents_status"), "documents", ["status"])


def downgrade() -> None:
    op.drop_index(op.f("ix_documents_status"), table_name="documents")
    op.drop_index(op.f("ix_documents_id"), table_name="documents")
    op.drop_table("documents")
