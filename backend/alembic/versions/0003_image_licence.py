"""Add licence column to medical_images

Revision ID: 0003_image_licence
Revises: 0002_medical_images
Created: 2026-08-11

WHY A COLUMN AND NOT A JSON KEY.
Figures ingested from PMC are redistributable because they come from the Open
Access Subset — but "open access" is not a single licence. CC-BY permits
derivatives with attribution; CC-BY-NC-ND permits neither commercial use nor
derivatives. A system that checks the licence at ingest time and then discards
the answer has a provenance *claim*, not a provenance *record*, and the
difference shows up exactly when someone asks whether a figure can be reused.

Nullable, because uploads have no upstream licence to record. That is a real
distinction from "we ingested this and did not check", which is why there is
no server_default filling the gap with a guess.

IDEMPOTENT, like 0001. These migrations run on every container start via
`alembic upgrade head`, and a half-applied schema from an interrupted start
must not wedge the next one.
"""
from alembic import op
import sqlalchemy as sa

revision = "0003_image_licence"
down_revision = "0002_medical_images"
branch_labels = None
depends_on = None


def _has_column(table: str, column: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table not in inspector.get_table_names():
        return False
    return any(c["name"] == column for c in inspector.get_columns(table))


def upgrade() -> None:
    if _has_column("medical_images", "licence"):
        return
    op.add_column(
        "medical_images",
        sa.Column("licence", sa.String(length=100), nullable=True),
    )


def downgrade() -> None:
    if _has_column("medical_images", "licence"):
        op.drop_column("medical_images", "licence")
