"""Add users table

Revision ID: 0005_users
Revises: 0004_reports
Created: 2026-08-12

Phase 6, Step 1. Until now every endpoint was open: anyone who could reach the
port could list every report and download every uploaded image.

⚠️  NO PLAINTEXT PASSWORD COLUMN EXISTS, AND NONE SHOULD BE ADDED.
`hashed_password` holds bcrypt output. There is no reset token and no recovery
email — each would be another credential path needing the same protection as
the password, and for operator-created accounts the recovery procedure is that
the operator runs the script again.

IDEMPOTENT, like 0001, 0003 and 0004 — these run on every container start.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005_users"
down_revision = "0004_reports"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if _has_table("users"):
        return

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),

        # 320 = the maximum length of an email address per RFC 5321
        # (64 local + @ + 255 domain). Stored lowercase; see the model.
        sa.Column("email", sa.String(length=320), nullable=False),

        # bcrypt output is 60 characters, but the column is wider so a future
        # move to argon2 does not need a migration under time pressure.
        sa.Column("hashed_password", sa.String(length=255), nullable=False),

        sa.Column("full_name", sa.String(length=200), nullable=True),

        # Deactivate rather than delete: a departed clinician's approvals must
        # stay attributable to them.
        sa.Column("is_active", sa.Boolean(), nullable=False,
                  server_default=sa.text("true")),
        sa.Column("is_admin", sa.Boolean(), nullable=False,
                  server_default=sa.text("false")),

        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),

        sa.PrimaryKeyConstraint("id"),
        # ⚠️  UNIQUENESS IS ENFORCED IN THE DATABASE, NOT ONLY IN CODE.
        # An application-level check races: two concurrent requests can both
        # see "no such user" and both insert. The constraint is the only thing
        # that actually holds.
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    op.create_index("ix_users_id", "users", ["id"])
    op.create_index("ix_users_email", "users", ["email"])


def downgrade() -> None:
    if _has_table("users"):
        op.drop_table("users")
