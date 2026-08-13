"""Add user ownership to reports and images

Revision ID: 0006_ownership
Revises: 0005_users
Created: 2026-08-12

Phase 6, Step 3. Until now every signed-in user could see every report and
every uploaded image — authentication without authorisation, which is a lock
on the front door of a building with no internal walls.

This is also what makes self-service registration safe. Without ownership, the
next person to sign up reads everything; with it, they land in an empty
workspace. The two must ship together, and they do.

════════════════════════════════════════════════════════════════════
⚠️  NULLABLE, AND EXISTING ROWS ARE NOT BACKFILLED
════════════════════════════════════════════════════════════════════
The tempting move is to assign every pre-auth row to the first registered
account. That would be convenient and it would be a lie: those reports were
created before the system knew who anyone was, so their author is genuinely
unknown.

An audit trail exists to record what actually happened. Inventing an
attribution to avoid a NULL is precisely the failure it is meant to prevent —
and it is worse than a gap, because a gap is visibly a gap while a fabricated
owner is indistinguishable from a real one.

So: user_id stays NULL for anything created before this migration, and those
rows report as unowned.

IDEMPOTENT, like every migration here — they run on each container start.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0006_ownership"
down_revision = "0005_users"
branch_labels = None
depends_on = None


def _has_column(table: str, column: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table not in inspector.get_table_names():
        return False
    return any(c["name"] == column for c in inspector.get_columns(table))


def upgrade() -> None:
    for table in ("reports", "medical_images"):
        if _has_column(table, "user_id"):
            continue

        op.add_column(
            table,
            sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        )

        # ⚠️  SET NULL, NEVER CASCADE.
        # Deleting a user must not delete their reports. Approvals have to
        # stay auditable after someone leaves — which is also why users are
        # deactivated rather than deleted in the first place.
        op.create_foreign_key(
            f"fk_{table}_user_id", table, "users",
            ["user_id"], ["id"], ondelete="SET NULL",
        )

        # Every list query now filters on this column.
        op.create_index(f"ix_{table}_user_id", table, ["user_id"])

    # reviewed_by becomes a real reference rather than free text.
    #
    # ⚠️  THE OLD STRING COLUMN IS KEPT.
    # It holds whatever was typed before authentication existed. Dropping it
    # would erase the only record of who claimed to have approved those
    # reports — thin evidence, but evidence, and the migration has no business
    # deciding it is worthless.
    if not _has_column("reports", "reviewed_by_user_id"):
        op.add_column(
            "reports",
            sa.Column("reviewed_by_user_id", postgresql.UUID(as_uuid=True),
                      nullable=True),
        )
        op.create_foreign_key(
            "fk_reports_reviewed_by", "reports", "users",
            ["reviewed_by_user_id"], ["id"], ondelete="SET NULL",
        )


def downgrade() -> None:
    if _has_column("reports", "reviewed_by_user_id"):
        op.drop_constraint("fk_reports_reviewed_by", "reports", type_="foreignkey")
        op.drop_column("reports", "reviewed_by_user_id")

    for table in ("reports", "medical_images"):
        if _has_column(table, "user_id"):
            op.drop_index(f"ix_{table}_user_id", table_name=table)
            op.drop_constraint(f"fk_{table}_user_id", table, type_="foreignkey")
            op.drop_column(table, "user_id")
